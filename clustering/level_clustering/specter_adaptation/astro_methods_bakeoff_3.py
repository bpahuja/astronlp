#!/usr/bin/env python3
"""
Astro Methods Bake-off + OOM-safe hyperparam search

- Train contrastive adapters on positive pairs for multiple base models (e.g., SPECTER2 vs AstroBERT).
- Keyword-masking augmentation to reduce reliance on giveaway terms.
- Side-by-side evaluation on val pairs (cosine positives vs random).
- Optional paragraph clustering (UMAP→HDBSCAN) to compare noise % and #clusters.
- OOM-safe hyperparameter search: each trial writes its own result files, frees CUDA memory between trials,
  and a final collation step summarizes all trials.

Requirements:
  pip install "sentence-transformers>=2.4.0" "transformers>=4.30" adapters torch \
              umap-learn hdbscan scikit-learn tqdm pandas

Examples:
  Bake-off:
    python astro_methods_bakeoff.py \
      --pairs_dir /path/to/pairs --out_root ./bakeoff_out \
      --bases allenai/specter2_base,adsabs/astroBERT \
      --lexicon_json /path/to/lexicon_v2.json \
      --epochs 6 --batch_size 128 --mask_prob 0.5 --device auto \
      --hf_cache /scratch/hf_cache

  Hyperparam search (OOM safe):
    python astro_methods_bakeoff.py \
      --mode hparam_search --base_model adsabs/astroBERT \
      --pairs_dir /path/to/pairs --out_root ./search_out \
      --lexicon_json /path/to/lexicon_v2.json \
      --search_trials 10 --search_epochs 2 \
      --hf_cache /scratch/hf_cache --device auto --enable_grad_ckpt
"""
import os, re, json, random, argparse, itertools, gc, traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models, losses, InputExample, util

import adapters
from adapters import AdapterConfig

# ---------------------- Utils ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_device(device_arg: str) -> str:
    if device_arg and device_arg.lower() != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s.strip())

def cleanup_cuda(*objs):
    """Move models to CPU (if possible), delete refs, collect, free CUDA cache."""
    for o in objs:
        try:
            # Attempt to move HF model to CPU
            if hasattr(o, "_first_module"):
                fm = o._first_module()
                if hasattr(fm, "auto_model"):
                    fm.auto_model.to("cpu")
        except Exception:
            pass
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

# ---------------------- Lexicon / masking ----------------------

def load_lexicon(lexicon_json: Optional[str]) -> List[re.Pattern]:
    if not lexicon_json:
        return []
    p = Path(lexicon_json)
    if not p.exists():
        print(f"[warn] lexicon file not found: {lexicon_json} (masking disabled)")
        return []
    try:
        lex = json.loads(Path(lexicon_json).read_text(encoding="utf-8"))
        pats = []
        for fam, lst in lex.items():
            if not lst: continue
            big = "(?:" + "|".join(lst) + ")"
            pats.append(re.compile(big, re.I))
        return pats
    except Exception as e:
        print(f"[warn] failed to load lexicon: {e}")
        return []

def mask_text(text: str, pat_list: List[re.Pattern], mask_prob: float, mask_token: str = "[MASK]", max_spans: int = 5) -> str:
    if not pat_list or mask_prob <= 0.0 or random.random() > mask_prob:
        return text
    spans = []
    for rgx in pat_list:
        for m in rgx.finditer(text):
            spans.append((m.start(), m.end()))
            if len(spans) >= max_spans:
                break
        if len(spans) >= max_spans:
            break
    if not spans:
        return text
    spans.sort()
    merged = []
    for s,e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    out = []
    cur = 0
    for s,e in merged:
        out.append(text[cur:s])
        out.append(mask_token)
        cur = e
    out.append(text[cur:])
    return "".join(out)

# ---------------------- Data loaders ----------------------

@dataclass
class PairItem:
    a: str
    b: str

class WindowedPairsDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, window_tokens=180, stride=30,
                 mask_patterns: List[re.Pattern] = None, mask_prob: float = 0.0):
        self.data: List[PairItem] = []
        self.tok = tokenizer
        self.W = window_tokens
        self.S = stride
        self.mask_patterns = mask_patterns or []
        self.mask_prob = mask_prob
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                self.data.append(PairItem(a=o["text_a"], b=o["text_b"]))

    def __len__(self):
        return len(self.data)

    def _sample_window(self, text: str) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.W:
            return text
        step = max(1, self.W - self.S)
        starts = list(range(0, max(1, len(ids)-self.W+1), step))
        s = random.choice(starts)
        sub = ids[s:s+self.W]
        return self.tok.decode(sub, skip_special_tokens=True)

    def __getitem__(self, idx):
        it = self.data[idx]
        ta = it.a; tb = it.b
        if self.mask_patterns and self.mask_prob > 0:
            ta = mask_text(ta, self.mask_patterns, self.mask_prob)
            tb = mask_text(tb, self.mask_patterns, self.mask_prob)
        ta = self._sample_window(ta)
        tb = self._sample_window(tb)
        return InputExample(texts=[ta, tb])

def build_dataloaders(pairs_dir: Path, tokenizer, batch_size: int, grad_accum: int,
                      window_tokens: int, stride: int, mask_patterns: List[re.Pattern], mask_prob: float,
                      num_workers: int, use_cuda: bool):
    train_path = pairs_dir / "pairs_train.jsonl"
    val_path   = pairs_dir / "pairs_val.jsonl"
    assert train_path.exists(), f"Missing {train_path}"

    pin = bool(use_cuda)
    pers = bool(use_cuda and num_workers > 0)
    per_step_bs = batch_size  # do not divide by grad_accum here

    train_ds = WindowedPairsDataset(train_path, tokenizer, window_tokens, stride, mask_patterns, mask_prob)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=per_step_bs, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pers
    )
    val_loader = None
    if val_path.exists():
        val_ds = WindowedPairsDataset(val_path, tokenizer, window_tokens, stride, mask_patterns=None, mask_prob=0.0)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=per_step_bs, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin, persistent_workers=pers
        )
    return train_loader, val_loader

# ---------------------- Model with adapter ----------------------

def build_st_with_adapter(base_id: str, adapter_name: str, adapter_type: str, reduction_factor: int,
                          max_seq_length: int, device: str, hf_cache: Optional[str]=None, local_files_only: bool=False,
                          enable_grad_ckpt: bool=False):
    try:
        word = models.Transformer(base_id, max_seq_length=max_seq_length, cache_dir=hf_cache, model_args={'cache_dir': hf_cache, 'local_files_only': local_files_only})
    except TypeError:
        word = models.Transformer(base_id, max_seq_length=max_seq_length)
    if enable_grad_ckpt:
        try:
            word.auto_model.gradient_checkpointing_enable()
            print("[grad-ckpt] Enabled gradient checkpointing on base model")
        except Exception:
            print("[grad-ckpt] Could not enable gradient checkpointing (ignored)")
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st = SentenceTransformer(modules=[word, pool], device=device)
    adapters.init(word.auto_model)
    cfg = AdapterConfig.load(adapter_type, reduction_factor=reduction_factor, non_linearity="swish")
    try:
        if adapter_name not in word.auto_model.adapters_config.adapters:
            word.auto_model.add_adapter(adapter_name, config=cfg)
    except Exception:
        try:
            word.auto_model.add_adapter(adapter_name, config=cfg)
        except Exception:
            pass
    word.auto_model.train_adapter(adapter_name)
    word.auto_model.set_active_adapters(adapter_name)
    return st

def save_adapter_from_st(st_model: SentenceTransformer, out_dir: Path, adapter_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    st_model.save(str(out_dir))
    first = st_model._first_module()
    first.auto_model.save_adapter(str(out_dir / "adapter"), adapter_name)

# ---------------------- Train / Eval ----------------------

def train_contrastive(st_model: SentenceTransformer, train_loader, epochs: int, lr: float, grad_accum: int, use_amp: bool=True):
    loss = losses.MultipleNegativesRankingLoss(st_model)
    steps_per_epoch = len(train_loader)
    total_steps = (steps_per_epoch * epochs) // max(1, grad_accum)
    warmup = max(1, int(0.1 * total_steps))
    try:
        st_model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=epochs,
            optimizer_params={"lr": lr},
            warmup_steps=warmup,
            use_amp=use_amp,
            gradient_accumulation_steps=max(1, grad_accum),
            show_progress_bar=True,
            checkpoint_path=None
        )
    except TypeError:
        print("[warn] gradient_accumulation_steps not supported by your sentence-transformers version. "
              "Falling back to no accumulation (increase --batch_size or upgrade ST).")
        st_model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=epochs,
            optimizer_params={"lr": lr},
            warmup_steps=warmup,
            use_amp=use_amp,
            show_progress_bar=True,
            checkpoint_path=None
        )

def eval_cosine(st_model: SentenceTransformer, pairs_jsonl: Path, batch_size: int = 64) -> Tuple[float, float]:
    rows = []
    with pairs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            rows.append((o["text_a"], o["text_b"]))
    if not rows:
        return float("nan"), float("nan")
    random.shuffle(rows)
    A = [a for a,b in rows]
    B = [b for a,b in rows]
    B_rand = random.sample(B, len(B))
    va = st_model.encode(A, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    vb = st_model.encode(B, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    vbr = st_model.encode(B_rand, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    pos = util.cos_sim(va, vb).diagonal().cpu().numpy().mean()
    rnd = util.cos_sim(va, vbr).diagonal().cpu().numpy().mean()
    return float(pos), float(rnd)

# ---------------------- Optional: clustering eval ----------------------

def clean_text(t: str) -> str:
    MATH = re.compile(r"(\$[^$]+\$)|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")
    CIT  = re.compile(r"\(([^()]*\d{4}[^()]*)\)|\[[0-9,\s;]+\]")
    URL  = re.compile(r"https?://\S+")
    FIG  = re.compile(r"^\s*(Figure|Table)\s+\d+", re.I)
    t = re.sub(MATH, " ", t or "")
    t = re.sub(CIT, " <CIT> ", t)
    t = re.sub(URL, " ", t)
    lines = [ln for ln in (t or "").splitlines() if not re.match(FIG, ln or "")]
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def read_method_paras(txt_dir: Path, sample: int = 0) -> pd.DataFrame:
    rows = []
    files = sorted(txt_dir.glob("*.txt"))
    rng = random.Random(42)
    if sample and sample < len(files):
        files = rng.sample(files, sample)
    for p in files:
        pid = p.stem
        raw = p.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"\n\s*\n", raw.strip())
        if len(parts) == 1:
            parts = [ln for ln in raw.splitlines() if ln.strip()]
        parts = [clean_text(x) for x in parts]
        parts = [x for x in parts if len(x) >= 200]
        for i, x in enumerate(parts):
            rows.append({"paper_id": pid, "para_id": f"{pid}::p{i:04d}", "text": x})
    df = pd.DataFrame(rows)
    return df

def encode_hier(st_model: SentenceTransformer, tokenizer, texts: List[str], window_tokens=180, stride=30, batch_size=64):
    embs = []
    for txt in tqdm(texts, desc="Encode (hierarchical)"):
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(ids) <= window_tokens:
            vec = st_model.encode([txt], convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)[0]
            embs.append(vec); continue
        chunks = []
        step = max(1, window_tokens - stride)
        for s in range(0, max(1, len(ids) - window_tokens + 1), step):
            sub = tokenizer.decode(ids[s:s+window_tokens], skip_special_tokens=True)
            chunks.append(sub)
        vecs = st_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        embs.append(vecs.mean(axis=0))
    return np.vstack(embs)

def clustering_eval(st_model: SentenceTransformer, base_id: str, paras_dir: Path,
                    out_dir: Path, sample_files: int, window_tokens: int, stride: int,
                    umap_dim: int, n_neighbors: int, min_cluster_size: int, min_samples: int,
                    hf_cache: Optional[str]=None, local_files_only: bool=False):
    try:
        import umap
        import hdbscan
    except Exception as e:
        print("[warn] clustering libs missing; skipping clustering eval:", e)
        return None

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, cache_dir=hf_cache, local_files_only=local_files_only)
    df = read_method_paras(paras_dir, sample=sample_files)
    if df.empty:
        return None
    embs = encode_hier(st_model, tokenizer, df["text"].tolist(), window_tokens, stride)

    reducer = umap.UMAP(n_components=umap_dim, n_neighbors=n_neighbors, min_dist=0.05, metric="cosine", random_state=42)
    Z = reducer.fit_transform(embs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(Z)
    n = len(labels); n_noise = int((labels < 0).sum())
    n_clusters = int(labels.max() + 1) if (labels >= 0).any() else 0
    noise_pct = 100.0 * n_noise / n if n else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"para_id": df["para_id"], "paper_id": df["paper_id"], "cluster": labels}).to_csv(out_dir / "paragraph_clusters.csv", index=False)
    summary = {"paragraphs": int(n), "clusters": int(n_clusters), "noise": int(n_noise), "noise_pct": float(noise_pct)}
    Path(out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

# ---------------------- Main routines ----------------------

def run_single_training(base_id: str, pairs_dir: Path, out_root: Path, lexicon_json: Optional[str], hf_cache: Optional[str], local_files_only: bool,
                        adapter_type: str, reduction_factor: int,
                        epochs: int, batch_size: int, grad_accum: int, lr: float,
                        window_tokens: int, stride: int, max_seq_length: int,
                        mask_prob: float, device: str, num_workers: int,
                        do_clustering: bool, paras_dir: Optional[Path],
                        cluster_sample: int, umap_dim: int, n_neighbors: int, min_cluster_size: int, min_samples: int,
                        enable_grad_ckpt: bool=False):
    set_seed(42)
    base_name = sanitize_name(base_id)
    out_dir = out_root / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, cache_dir=hf_cache, local_files_only=local_files_only)
    mask_patterns = load_lexicon(lexicon_json)

    use_cuda = device.startswith("cuda")
    train_loader, val_loader = build_dataloaders(
        pairs_dir, tokenizer, batch_size, grad_accum, window_tokens, stride,
        mask_patterns, mask_prob, num_workers, use_cuda
    )

    adapter_name = "astro_methods"
    st = build_st_with_adapter(base_id, adapter_name, adapter_type, reduction_factor, max_seq_length, device,
                               hf_cache=hf_cache, local_files_only=local_files_only, enable_grad_ckpt=enable_grad_ckpt)

    print(f"[train] base={base_id} adapter={adapter_type} r={reduction_factor} lr={lr} epochs={epochs} mask_prob={mask_prob}")
    train_contrastive(st, train_loader, epochs=epochs, lr=lr, grad_accum=grad_accum, use_amp=True)

    save_adapter_from_st(st, out_dir, adapter_name)

    val_path = pairs_dir / "pairs_val.jsonl"
    pos, rnd = eval_cosine(st, val_path)
    metrics = {"pos_mean": pos, "rand_mean": rnd, "gap": (pos - rnd)}
    Path(out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] {base_name}: pos={pos:.4f} rand={rnd:.4f} gap={pos-rnd:.4f}")

    cluster_summary = None
    if do_clustering and paras_dir and paras_dir.exists():
        print("[cluster] running paragraph clustering eval...")
        cluster_summary = clustering_eval(
            st, base_id, paras_dir, out_dir / "clustering_eval",
            sample_files=cluster_sample, window_tokens=window_tokens, stride=stride,
            umap_dim=umap_dim, n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            hf_cache=hf_cache, local_files_only=local_files_only
        )
        if cluster_summary:
            print(f"[cluster] noise={cluster_summary['noise_pct']:.1f}% clusters={cluster_summary['clusters']} paragraphs={cluster_summary['paragraphs']}")

    # Free memory for next run
    cleanup_cuda(st, train_loader, val_loader, tokenizer)
    return {"base": base_id, **metrics, "cluster": cluster_summary or {}}

def bakeoff(bases: List[str], **kwargs):
    results = []
    for base in bases:
        r = run_single_training(base_id=base, **kwargs)
        results.append(r)
    out_root = kwargs["out_root"]
    df = pd.DataFrame([{
        "base": r["base"],
        "pos_mean": r["pos_mean"],
        "rand_mean": r["rand_mean"],
        "gap": r["gap"],
        "cluster_noise_pct": r["cluster"].get("noise_pct", None),
        "cluster_n_clusters": r["cluster"].get("clusters", None),
        "cluster_paragraphs": r["cluster"].get("paragraphs", None),
    } for r in results])
    df.to_csv(out_root / "bakeoff_summary.csv", index=False)
    print("\n=== Bake-off summary ===")
    print(df.to_string(index=False))
    best = max(results, key=lambda x: (x["gap"], -x["cluster"].get("noise_pct", 1e9)))
    Path(out_root / "winner.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nWinner: {best['base']} (gap={best['gap']:.4f})")
    return best["base"]

# ---------------------- OOM-safe hyperparameter search ----------------------

def hparam_search(base_model: str, pairs_dir: Path, out_root: Path, lexicon_json: Optional[str],
                  device: str, num_workers: int, search_trials: int, search_epochs: int,
                  window_tokens: int, stride: int, max_seq_length: int,
                  hf_cache: Optional[str]=None, local_files_only: bool=False, enable_grad_ckpt: bool=False):
    # Ensure per-trial dir
    trials_dir = out_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, cache_dir=hf_cache, local_files_only=local_files_only)
    mask_patterns = load_lexicon(lexicon_json)
    use_cuda = device.startswith("cuda")

    # Smaller default batch sizes to reduce OOM risk
    adapter_types = ["pfeiffer", "houlsby"]
    reduction_factors = [8, 12, 16]
    lrs = [1e-4, 5e-5]
    mask_probs = [0.0, 0.5]
    batch_sizes = [64]  # smaller for safety
    grad_accums = [1, 2]

    combos = list(itertools.product(adapter_types, reduction_factors, lrs, mask_probs, batch_sizes, grad_accums))
    random.seed(42); random.shuffle(combos)
    combos = combos[:max(1, search_trials)]

    results = []
    for i, (atype, rf, lr, mprob, bs, ga) in enumerate(combos, 1):
        trial_name = f"t{i:02d}_{atype}_r{rf}_lr{lr}_m{mprob}_bs{bs}_ga{ga}"
        tdir = trials_dir / trial_name
        tdir.mkdir(parents=True, exist_ok=True)
        cfg = {"adapter_type": atype, "reduction": rf, "lr": lr, "mask_prob": mprob,
               "batch_size": bs, "grad_accum": ga, "base_model": base_model,
               "window_tokens": window_tokens, "stride": stride, "max_seq_length": max_seq_length,
               "epochs": search_epochs}
        Path(tdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"\n[search {i}/{len(combos)}] {trial_name}")

        # Build loaders fresh per trial (so they can be GC'ed)
        try:
            train_loader, _ = build_dataloaders(
                pairs_dir, tokenizer, bs, ga, window_tokens, stride, mask_patterns, mprob, num_workers, use_cuda
            )
            adapter_name = f"search_{atype}_r{rf}"
            st = build_st_with_adapter(base_model, adapter_name, atype, rf, max_seq_length, device,
                                       hf_cache=hf_cache, local_files_only=local_files_only,
                                       enable_grad_ckpt=enable_grad_ckpt)
            train_contrastive(st, train_loader, epochs=search_epochs, lr=lr, grad_accum=ga, use_amp=True)
            val_path = pairs_dir / "pairs_val.jsonl"
            pos, rnd = eval_cosine(st, val_path)
            gap = pos - rnd
            res = {"status": "ok", "pos": pos, "rand": rnd, "gap": gap}
            Path(tdir / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[trial] pos={pos:.4f} rand={rnd:.4f} gap={gap:.4f}")
            results.append({**cfg, **res})
        except torch.cuda.OutOfMemoryError as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            print(f"[OOM] {trial_name}: {msg}")
            Path(tdir / "metrics.json").write_text(json.dumps({"status": "oom", "error": msg}, indent=2), encoding="utf-8")
        except Exception as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            print(f"[error] {trial_name}: {msg}")
            Path(tdir / "metrics.json").write_text(json.dumps({"status": "error", "error": msg}, indent=2), encoding="utf-8")
        finally:
            # Free CUDA memory aggressively
            cleanup_cuda(locals().get("st"), locals().get("train_loader"))
            print("[cleanup] Freed CUDA cache")

    # Collate all trial files (even if the run was interrupted)
    rows = []
    for mjson in trials_dir.glob("*/metrics.json"):
        cfg_path = mjson.parent / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            met = json.loads(mjson.read_text(encoding="utf-8"))
            rows.append({**cfg, **met})
        except Exception:
            pass
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "hparam_results.csv", index=False)
        ok = df[df["status"]=="ok"]
        if not ok.empty:
            best = ok.loc[ok["gap"].idxmax()].to_dict()
            Path(out_root / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
            print("\n=== Hyperparam search (best by gap) ===")
            print({k: best[k] for k in ["adapter_type","reduction","lr","mask_prob","batch_size","grad_accum","gap"]})
        else:
            print("\nNo successful trials to summarize (all OOM or errors).")
    else:
        print("\nNo trial result files found to collate.")

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bakeoff", choices=["bakeoff", "hparam_search"])
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--bases", default="allenai/specter2_base,adsabs/astroBERT",
                    help="Comma-separated base model IDs (bakeoff mode)")
    ap.add_argument("--base_model", default=None, help="Single base model (for hparam_search mode)")
    ap.add_argument("--lexicon_json", default=None)

    ap.add_argument("--adapter_type", default="pfeiffer", choices=["pfeiffer","houlsby"])
    ap.add_argument("--reduction_factor", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--window_tokens", type=int, default=180)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--mask_prob", type=float, default=0.5)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=2)

    # HF cache & offline
    ap.add_argument("--hf_cache", default=None, help="Directory to use for Hugging Face cache")
    ap.add_argument("--local_files_only", action="store_true", help="Only load models from local cache (no download)")

    # Clustering eval
    ap.add_argument("--paras_dir", default=None)
    ap.add_argument("--cluster_sample", type=int, default=0, help="Sample N files (0=all)")
    ap.add_argument("--umap_dim", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=150)
    ap.add_argument("--min_cluster_size", type=int, default=10)
    ap.add_argument("--min_samples", type=int, default=2)

    # Hyperparam search
    ap.add_argument("--search_trials", type=int, default=6)
    ap.add_argument("--search_epochs", type=int, default=2)
    ap.add_argument("--enable_grad_ckpt", action="store_true", help="Enable gradient checkpointing to reduce memory")

    args = ap.parse_args()

    # Memory fragmentation mitigation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.hf_cache:
        os.makedirs(args.hf_cache, exist_ok=True)
        os.environ['TRANSFORMERS_CACHE'] = args.hf_cache
        os.environ['HF_HOME'] = args.hf_cache
        print(f"[hf-cache] Using cache dir: {args.hf_cache}")

    set_seed(42)
    device = parse_device(args.device)
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"[device] Using {device}")
    else:
        print("[device] Using CPU")

    pairs_dir = Path(args.pairs_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "bakeoff":
        bases = [b.strip() for b in args.bases.split(",") if b.strip()]
        do_cluster = bool(args.paras_dir)
        paras_dir = Path(args.paras_dir) if args.paras_dir else None
        bakeoff(
            bases=bases,
            pairs_dir=pairs_dir,
            out_root=out_root,
            lexicon_json=args.lexicon_json,
            hf_cache=args.hf_cache,
            local_files_only=args.local_files_only,
            adapter_type=args.adapter_type,
            reduction_factor=args.reduction_factor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            window_tokens=args.window_tokens,
            stride=args.stride,
            max_seq_length=args.max_seq_length,
            mask_prob=args.mask_prob,
            device=device,
            num_workers=args.num_workers,
            do_clustering=do_cluster,
            paras_dir=paras_dir,
            cluster_sample=args.cluster_sample,
            umap_dim=args.umap_dim,
            n_neighbors=args.n_neighbors,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            enable_grad_ckpt=args.enable_grad_ckpt
        )
    else:
        base = args.base_model or "allenai/specter2_base"
        hparam_search(
            base_model=base,
            pairs_dir=pairs_dir,
            out_root=out_root,
            lexicon_json=args.lexicon_json,
            device=device,
            num_workers=args.num_workers,
            search_trials=args.search_trials,
            search_epochs=args.search_epochs,
            window_tokens=args.window_tokens,
            stride=args.stride,
            max_seq_length=args.max_seq_length,
            hf_cache=args.hf_cache,
            local_files_only=args.local_files_only,
            enable_grad_ckpt=args.enable_grad_ckpt
        )

if __name__ == "__main__":
    main()
