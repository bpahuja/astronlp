import torch
from torch.utils.data import DataLoader
from bert_classifier_trainer import BERTMetaClassifierTrainer, ParagraphDataset
import os
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_bootstrap_checkpoint(checkpoint_dir, iteration, trainer, train_df, val_df, unlabelled_df):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save dataframes
    train_df.to_json(os.path.join(checkpoint_dir, f"train_iter{iteration}.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(checkpoint_dir, f"val_iter{iteration}.jsonl"), orient="records", lines=True)
    unlabelled_df.to_json(os.path.join(checkpoint_dir, f"unlabelled_iter{iteration}.jsonl"), orient="records", lines=True)
    # Save model state dict
    model_path = os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")
    torch.save(trainer.model.state_dict(), model_path)
    # Save checkpoint meta info
    with open(os.path.join(checkpoint_dir, "bootstrap_state.json"), "w") as f:
        json.dump({
            "last_iteration": iteration,
            "model_path": model_path,
            "train_path": f"train_iter{iteration}.jsonl",
            "val_path": f"val_iter{iteration}.jsonl",
            "unlabelled_path": f"unlabelled_iter{iteration}.jsonl"
        }, f)
    print(f"[Checkpoint] Saved bootstrapping state at iteration {iteration}")

def load_bootstrap_checkpoint(checkpoint_dir, trainer):
    # Read meta info
    state_file = os.path.join(checkpoint_dir, "bootstrap_state.json")
    if not os.path.exists(state_file):
        print("[Checkpoint] No checkpoint found, starting fresh.")
        return None
    with open(state_file, "r") as f:
        state = json.load(f)
    iteration = state["last_iteration"]
    train_df = pd.read_json(os.path.join(checkpoint_dir, state["train_path"]), lines=True)
    val_df = pd.read_json(os.path.join(checkpoint_dir, state["val_path"]), lines=True)
    unlabelled_df = pd.read_json(os.path.join(checkpoint_dir, state["unlabelled_path"]), lines=True)
    # Load model state dict
    model_path = os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")
    trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    print(f"[Checkpoint] Resuming from iteration {iteration}")
    return iteration, train_df, val_df, unlabelled_df

def filter_by_confidence(logits, threshold=0.97):
    probs = torch.softmax(logits, dim=1)                  # [N, num_classes]
    confidence, pred = torch.max(probs, dim=1)             # [N], [N]
    mask = confidence > threshold                          # [N] boolean
    return pred, mask

def robust_bootstrap_with_resume(
    trainer,
    train_df,
    val_df,
    unlabelled_df,
    output_dir,
    confidence_threshold=0.97,
    num_iterations=5,
    batch_size=32
):
    # Check for existing checkpoint
    checkpoint_dir = os.path.join(output_dir, "bootstrap_checkpoints")
    checkpoint = load_bootstrap_checkpoint(checkpoint_dir, trainer)
    if checkpoint:
        start_iter, train_df, val_df, unlabelled_df = checkpoint
        start_iter += 1  # Start at next round
    else:
        start_iter = 1

    for it in range(start_iter, num_iterations+1):
        print(f"\n=== Bootstrapping Iteration {it} ===")

        if it == 1:
            # Using pretrained model for the first iteration
            print("Using pretrained model for the first iteration.")
            trainer.model.train()  # Ensure model is in training mode
        else:
            # Load the model state from the last iteration
            print(f"Loading model state from iteration {it-1}")
            model_path = os.path.join(checkpoint_dir, f"model_iter{it-1}.pt")
            trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
            trainer.model.train()

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Unlabelled: {len(unlabelled_df)}")

        if it > 1:
            # 1. Train on labelled set
            print("Training on labelled data...")
            train_on_df(trainer, train_df, val_df)

        # 2. Predict on unlabelled set
        print("Predicting logits on unlabelled set...")
        logits = predict_logits_on_paragraphs(trainer, unlabelled_df, batch_size=batch_size)

        # 3. Select high-confidence pseudo-labels
        pseudo_preds, mask = filter_by_confidence(logits, confidence_threshold)
        mask = mask.cpu().numpy()
        num_new = mask.sum()
        print(f"  Found {num_new} high-confidence pseudo-labels")
        if num_new == 0:
            print("  No more pseudo-labels to add. Stopping.")
            break

        # 4. Add to train_df
        add_df = unlabelled_df[mask].copy()
        add_df["label"] = pseudo_preds[mask].cpu().numpy()

        # 5. Remove from unlabelled_df
        unlabelled_df = unlabelled_df[~mask].reset_index(drop=True)

        # 6. Expand train set
        train_df = pd.concat([train_df, add_df], ignore_index=True)

        # 7. Save checkpoint (for resume)
        save_bootstrap_checkpoint(checkpoint_dir, it, trainer, train_df, val_df, unlabelled_df)

    print("Bootstrapping complete.")
    return trainer


def train_on_df(trainer, train_df, val_df):
    train_loader, val_loader = trainer.create_dataloaders(train_df, val_df)
    best_f1 = 0
    for epoch in tqdm((trainer.config['epochs'])):
        trainer.train_epoch(train_loader)
        _, val_f1, *_ = trainer.evaluate(val_loader)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = trainer.model.state_dict()
    trainer.model.load_state_dict(best_state)


def collate_fn(batch):
    """
    Collate function for batching samples with variable chunk lengths.
    Returns lists: input_ids_list, attention_mask_list, paragraph_sizes, labels
    """
    input_ids_list = []
    attention_mask_list = []
    paragraph_sizes = []
    labels = []
    for chunks, label in batch:
        input_ids_list.extend(chunks)
        attention_mask_list.extend([torch.ones_like(chunk) for chunk in chunks])
        paragraph_sizes.append(len(chunks))
        labels.append(label)
    labels = torch.stack(labels)
    return input_ids_list, attention_mask_list, paragraph_sizes, labels

def inference_collate_fn(batch):
    """
    Collate function for inference (no labels)
    """
    input_ids_list = []
    attention_mask_list = []
    paragraph_sizes = []
    for chunks in batch:
        input_ids_list.extend(chunks)
        attention_mask_list.extend([torch.ones_like(chunk) for chunk in chunks])
        paragraph_sizes.append(len(chunks))
    return input_ids_list, attention_mask_list, paragraph_sizes

def predict_logits_on_paragraphs(trainer, unlabelled_df, batch_size=32):
    """
    Batched inference using ParagraphDataset for unlabelled data.
    Returns: logits tensor [N, num_classes]
    """
    texts = unlabelled_df["input_text"].tolist()
    # Dummy labels (not used, but ParagraphDataset expects them)
    dummy_labels = [0] * len(texts)
    dataset = ParagraphDataset(texts, dummy_labels, trainer.tokenizer, max_length=trainer.config['max_length'])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: [b[0] for b in batch], pin_memory=True)

    logits_list = []
    trainer.model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch in tqdm(loader, desc="Inferencing (bootstrapping)"):
                # batch is a list of lists of chunk tensors
                input_ids_list, attention_mask_list, paragraph_sizes = inference_collate_fn(batch)
                input_ids_list = [x.to(trainer.device) for x in input_ids_list]
                attention_mask_list = [x.to(trainer.device) for x in attention_mask_list]
                logits = trainer.model(input_ids_list, attention_mask_list, paragraph_sizes)
                logits_list.append(logits.cpu())
                # Clear cache every 100 batches to avoid OOM
                if len(logits_list) % 100 == 0:
                    torch.cuda.empty_cache()
    return torch.cat(logits_list, dim=0)


if __name__ == "__main__":

    # If resuming, always use load_trained_model
    model_dir = "/vol/bitbucket/bp824/data/bert-classifier-final/output/model_20250621_125159"
    trainer, _ = BERTMetaClassifierTrainer.load_trained_model(model_dir)

    train_df, val_df = trainer.preprocess_data("data/methodolodgy_labels/labeled_paragraphs.jsonl")
    unlabelled_df = pd.read_json("data/unlabelled_methodology/unlabelled_paragraphs.jsonl", lines=True)

    def preprocess_row(row):
        paragraph = row["paragraph"]
        clean_paragraph = paragraph.replace("[MATH_tex=", "[MATH]").split("]")[-1]
        return f"{row['section']} - {clean_paragraph}"

    with ThreadPoolExecutor(max_workers=32) as executor:
        unlabelled_df["input_text"] = list(executor.map(preprocess_row, unlabelled_df.to_dict("records")))

    output_dir = "./bootstrap_output"
    robust_bootstrap_with_resume(
        trainer,
        train_df,
        val_df,
        unlabelled_df,
        output_dir=output_dir,
        confidence_threshold=0.95,
        num_iterations=5,
        batch_size=64
    )