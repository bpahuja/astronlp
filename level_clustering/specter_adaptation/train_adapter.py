#!/usr/bin/env python3
import os, json, random, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models, losses, InputExample

random.seed(42)
DEFAULT_BASE = "adsabs/astroBERT"
ADAPTER_NAME = "astro_methods_astrobert"

@dataclass
class PairItem:
    a: str
    b: str

class WindowedPairsDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, window_tokens=180, stride=30):
        self.data: List[PairItem] = []
        self.tok = tokenizer
        self.window_tokens = window_tokens
        self.stride = stride
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                o = json.loads(line)
                self.data.append(PairItem(a=o["text_a"], b=o["text_b"]))

    def __len__(self):
        return len(self.data)

    def _sample_window(self, text: str) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        W = self.window_tokens
        S = self.stride
        if len(ids) <= W:
            return text
        step = max(1, W - S)
        starts = list(range(0, max(1, len(ids)-W+1), step))
        s = random.choice(starts)
        sub = ids[s:s+W]
        return self.tok.decode(sub, skip_special_tokens=True)

    def __getitem__(self, idx):
        item = self.data[idx]
        ta = self._sample_window(item.a)
        tb = self._sample_window(item.b)
        return InputExample(texts=[ta, tb])

def build_model(base_id: str, max_seq_length: int, device: str):
    from sentence_transformers import SentenceTransformer, models
    import adapters
    from adapters import AdapterConfig

    word = models.Transformer(base_id, max_seq_length=max_seq_length)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[word, pool], device=device)

    # Init adapters on the underlying HF model, then add + activate Pfeiffer adapter
    adapters.init(word.auto_model)
    cfg = AdapterConfig.load("pfeiffer", reduction_factor=12)
    word.auto_model.add_adapter("astro_methods_astrobert", config=cfg)
    word.auto_model.train_adapter("astro_methods_astrobert")
    word.auto_model.set_active_adapters("astro_methods_astrobert")
    return model

def load_dataloaders(pairs_dir: Path, tokenizer, batch_size: int, window_tokens: int, stride: int,
                     use_cuda: bool, num_workers: int):
    train_path = pairs_dir / "pairs_train.jsonl"
    val_path   = pairs_dir / "pairs_val.jsonl"
    test_path  = pairs_dir / "pairs_test.jsonl"
    assert train_path.exists(), f"Missing {train_path}"

    pin_memory = bool(use_cuda)
    persistent_workers = bool(use_cuda and num_workers > 0)

    train_ds = WindowedPairsDataset(train_path, tokenizer, window_tokens, stride)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    val_loader = None
    if val_path.exists():
        val_loader = DataLoader(
            WindowedPairsDataset(val_path, tokenizer, window_tokens, stride),
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
    test_loader = None
    if test_path.exists():
        test_loader = DataLoader(
            WindowedPairsDataset(test_path, tokenizer, window_tokens, stride),
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
    return train_loader, val_loader, test_loader

def parse_device(device_arg: str) -> str:
    if device_arg and device_arg.lower() != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default=DEFAULT_BASE)

    # CUDA / performance
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 | cuda:1 ...")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--window_tokens", type=int, default=180)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (set 0 on Windows)")

    # AMP/mixed precision
    ap.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    args = ap.parse_args()

    device = parse_device(args.device)
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # speed up on fixed-size batches
        # respect CUDA_VISIBLE_DEVICES if set; otherwise, use user-provided device
        print(f"Using CUDA device: {device}")
    else:
        print("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    train_loader, val_loader, test_loader = load_dataloaders(
        Path(args.pairs_dir), tokenizer,
        batch_size=max(1, args.batch_size // max(1, args.grad_accum)),
        window_tokens=args.window_tokens, stride=args.stride,
        use_cuda=use_cuda, num_workers=args.num_workers
    )

    model = build_model(args.base_model, args.max_seq_length, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Sentence-Transformers supports AMP with use_amp=True
    use_amp = not args.no_amp

    # Warmup proportional to total optimization steps
    steps_per_epoch = len(train_loader)
    total_optim_steps = (steps_per_epoch * args.epochs) // max(1, args.grad_accum)
    warmup_steps = max(1, int(0.1 * total_optim_steps))

    # If doing gradient accumulation, wrap the loss to scale appropriately
    # Sentence-Transformers handles batches internally; we mimic larger batch via grad_accum
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=warmup_steps,
        use_amp=use_amp,
        checkpoint_path=None,
        # gradient_accumulation_steps=max(1, args.grad_accum),
        show_progress_bar=True
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    # Save adapter separately
    model._first_module().auto_model.save_adapter(str(out / "adapter"), ADAPTER_NAME)
    print(f"Saved model and adapter to: {out}")

if __name__ == "__main__":
    main()
