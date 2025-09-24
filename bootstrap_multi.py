from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import os
import json
import pandas as pd
from tqdm import tqdm
import time
import math
import argparse  # For command-line arguments

import lmdb
import pickle
import re
from datetime import datetime

# --- FIX for "Too many open files" error ---
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# --- 1. The Correct and OPTIMIZED Model Definition ---
class BERTMetaClassifier(nn.Module):
    def __init__(self, model_name, cache_path, hidden_dim=768, meta_hidden=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_path)
        self.lstm = nn.LSTM(hidden_dim, meta_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(meta_hidden * 2, num_classes)
        self.pad_token_id = self.bert.config.pad_token_id

    def forward(self, input_ids, attention_mask, paragraph_sizes):
        # input_ids: [total_chunks, seq_len]
        # attention_mask: [total_chunks, seq_len]
        # paragraph_sizes: list of ints (chunks per paragraph)

        # 1. Get CLS embeddings for all chunks in the batch at once
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :]  # [total_chunks, hidden_dim]

        # 2. Split the flat list of chunk embeddings back into a list of paragraphs
        # This is a list of tensors, where each tensor is a paragraph
        paragraph_chunk_embeddings = torch.split(cls_embeddings, paragraph_sizes)

        # 3. Pad the PARAGRAPHS so they all have the same number of chunks
        # This creates a single 3D tensor for the entire batch.
        # Shape: [batch_size, max_chunks_in_batch, hidden_dim]
        padded_paragraphs = pad_sequence(paragraph_chunk_embeddings, batch_first=True, padding_value=0.0)

        # 4. Pass the entire batch of paragraphs to the LSTM at once
        # The Python for loop is now gone!
        lstm_out, _ = self.lstm(padded_paragraphs)  # Shape: [batch_size, max_chunks, meta_hidden * 2]

        # 5. Average pool the LSTM outputs for each paragraph
        # We need to ignore the padded chunks during pooling
        # Create a mask for the paragraphs based on their original chunk counts
        paragraph_mask = torch.arange(padded_paragraphs.size(1), device=input_ids.device)[None, :] < torch.tensor(
            paragraph_sizes, device=input_ids.device)[:, None]

        # Expand mask to match lstm_out dimensions for masked averaging
        paragraph_mask = paragraph_mask.unsqueeze(-1).expand_as(lstm_out)

        # Perform masked averaging
        sum_lstm_out = torch.sum(lstm_out * paragraph_mask, dim=1)
        non_pad_elements = paragraph_mask.sum(dim=1)
        # Avoid division by zero for empty sequences
        non_pad_elements = torch.clamp(non_pad_elements, min=1e-9)

        pooled = sum_lstm_out / non_pad_elements
        pooled = self.dropout(pooled)  # Shape: [batch_size, meta_hidden * 2]

        return self.classifier(pooled)

# --- 2. Data Handling Components ---
def clean_math_tags(paragraph):
    return re.sub(r'\[MATH_tex=.*?\]', '[MATH]', paragraph)

def robust_jsonl_reader(path):
    """
    Reads a JSONL file line by line, skipping empty or malformed lines.
    """
    records = []
    print(f"Robustly reading JSONL file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Skip empty or whitespace-only lines
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {i+1} in {path}")
    if not records:
        raise ValueError(f"No valid JSON objects found in {path}. The file might be empty or completely malformed.")
    return pd.DataFrame(records)


class LMDBParagraphDataset(Dataset):
    def __init__(self, df, lmdb_path, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.lmdb_path = lmdb_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, idx):
        if self.env is None: self._init_db()
        row = self.df.iloc[idx]
        paper_id = row["paper_id"]
        section_idx = row["section_index"]
        para_idx = row["paragraph_index"]

        # *** FIX: Default to label 0 if the 'label' column is missing (for inference) ***
        label = row.get("label", 0)

        if isinstance(label, str):
            if label == 'methodology':
                label = 1
            else:
                label = 0

        key_str = f"{paper_id}|{section_idx}|{para_idx}"
        key_bytes = key_str.encode('utf-8')
        with self.env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
        if value_bytes:
            chunk_ids = pickle.loads(value_bytes)
        else:
            paragraph_text = clean_math_tags(row.get("paragraph", ""))
            chunk_ids = self._chunk_paragraph_text(paragraph_text)
        chunks = [torch.tensor(c, dtype=torch.long) for c in chunk_ids]
        return chunks, torch.tensor(label, dtype=torch.long)

    def _chunk_paragraph_text(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [[self.tokenizer.cls_token_id] + tokens[i:i + self.max_length - 2] + [self.tokenizer.sep_token_id] for
                  i in range(0, len(tokens), self.max_length - 2)]
        return chunks if chunks else [[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    all_chunks, all_attn, paragraph_sizes, labels = [], [], [], []
    for chunks, label in batch:
        all_chunks.extend(chunks)
        all_attn.extend([torch.ones_like(c) for c in chunks])
        paragraph_sizes.append(len(chunks))
        labels.append(label)
    input_ids = pad_sequence(all_chunks, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(all_attn, batch_first=True, padding_value=0)
    return input_ids, attention_mask, paragraph_sizes, torch.stack(labels)

def single_sample_collate_fn(batch):
    return batch[0]
# --- 3. The Integrated Trainer Class ---
class BERTMetaClassifierTrainer:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', cache_path='/vol/bitbucket/bp824/hf_models/scibert', output_dir='./output', config=None):
        self.model_name, self.cache_path, self.output_dir = model_name, cache_path, output_dir
        self.config = {'learning_rate': 1e-5, 'batch_size': 4, 'meta_hidden': 256, 'dropout': 0.2, 'epochs': 5, 'optimizer': 'adamw', 'max_length': 512, 'num_classes': 2, 'test_size': 0.2, 'random_state': 42}
        if config: self.config.update(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.model, self.tokenizer, self.optimizer = None, None, None
        self.scaler = GradScaler()
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Trainer initialized. Device: {self.device}")

    def load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_path, use_fast=True)
        self.model = BERTMetaClassifier(model_name=self.model_name, cache_path=self.cache_path, hidden_dim=768, meta_hidden=self.config['meta_hidden'], dropout=self.config['dropout'], num_classes=self.config['num_classes']).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

    def preprocess_data(self, data_path):
        print(f"Preprocessing data from: {data_path}")
        # *** FIX: Use the robust reader instead of pd.read_json directly ***
        df = pd.read_json(data_path, lines=True)
        df["label"] = df["label"].map({"methodology": 1, "not_methodology": 0})
        methodology_df, non_methodology_df = df[df['label'] == 1], df[df['label'] == 0]
        min_count = min(len(methodology_df), len(non_methodology_df))
        balanced_df = pd.concat([methodology_df.sample(n=min_count, random_state=self.config['random_state']), non_methodology_df.sample(n=min_count, random_state=self.config['random_state'])]).sample(frac=1, random_state=self.config['random_state']).reset_index(drop=True)
        train_df, test_df = train_test_split(balanced_df, test_size=self.config['test_size'], stratify=balanced_df["label"], random_state=self.config['random_state'])
        return train_df, test_df

    def create_dataloaders(self, train_df, test_df, lmdb_path):
        print("Creating LMDB-aware dataloaders...")
        train_dataset = LMDBParagraphDataset(train_df, lmdb_path, self.tokenizer, self.config['max_length'])
        test_dataset = LMDBParagraphDataset(test_df, lmdb_path, self.tokenizer, self.config['max_length'])
        train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=single_sample_collate_fn, num_workers=2)
        return train_dataloader, test_dataloader

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training Epoch"):
            input_ids, attention_mask, paragraph_sizes, labels = batch
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                outputs = self.model(input_ids, attention_mask, paragraph_sizes)
                loss = F.cross_entropy(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for chunks, label in tqdm(dataloader, desc="Evaluating"):
                input_ids = pad_sequence(chunks, batch_first=True, padding_value=0).to(self.device)
                attention_mask = (input_ids != 0).long().to(self.device)
                outputs = self.model(input_ids, attention_mask, [len(chunks)])
                all_preds.append(torch.argmax(outputs, dim=1).cpu())
                all_labels.append(label)
        all_preds, all_labels = torch.cat(all_preds), torch.tensor(all_labels)
        return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

    @classmethod
    def load_trained_model(cls, model_dir):
        info_path = os.path.join(model_dir, "model_info.json")
        with open(info_path, 'r') as f: model_info = json.load(f)
        trainer = cls(model_name=model_info['model_name'], cache_path=model_info.get('cache_path', ''), config=model_info['model_config'])
        trainer.load_model_and_tokenizer()
        trainer.model.load_state_dict(torch.load(model_info['model_path'], map_location=trainer.device))
        print(f"Model loaded from: {model_dir}")
        return trainer, model_info['model_config']


def save_bootstrap_checkpoint(checkpoint_dir, iteration, train_df, val_df, unlabelled_df, trainer):
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_df.to_json(os.path.join(checkpoint_dir, f"train_iter{iteration}.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(checkpoint_dir, f"val_iter{iteration}.jsonl"), orient="records", lines=True)
    unlabelled_df.to_json(os.path.join(checkpoint_dir, f"unlabelled_iter{iteration}.jsonl"), orient="records", lines=True)
    torch.save(trainer.model.state_dict(), os.path.join(checkpoint_dir, f"model_iter{iteration}.pt"))
    with open(os.path.join(checkpoint_dir, "bootstrap_state.json"), "w") as f: json.dump({"last_iteration": iteration}, f)
    print(f"[Checkpoint] Saved state for iteration {iteration}")

def load_bootstrap_checkpoint(checkpoint_dir, trainer):
    state_file = os.path.join(checkpoint_dir, "bootstrap_state.json")
    if not os.path.exists(state_file): return None, None, None, None
    with open(state_file, "r") as f: iteration = json.load(f)["last_iteration"]
    train_path, val_path, unlabelled_path, model_path = [os.path.join(checkpoint_dir, f"{name}_iter{iteration}.jsonl") for name in ["train", "val", "unlabelled"]] + [os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")]
    if not all(os.path.exists(p) for p in [train_path, val_path, unlabelled_path, model_path]): return None, None, None, None
    print(f"[Checkpoint] Resuming from iteration {iteration}")
    trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    return iteration, pd.read_json(train_path, lines=True), pd.read_json(val_path, lines=True), unlabelled_path

def filter_by_confidence(logits, max_new_labels, min_new_labels, confidence_threshold):
    probs = torch.softmax(logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    candidate_mask = confidence > confidence_threshold
    candidate_indices = torch.where(candidate_mask)[0]

    if len(candidate_indices) < min_new_labels:
        print(
            f"  Found only {len(candidate_indices)} new labels, which is below the minimum of {min_new_labels}. Stopping.")
        return None, False

    if len(candidate_indices) > max_new_labels:
        print(f"Found {len(candidate_indices)} candidates, capping to top {max_new_labels}.")
        candidate_confidences = confidence[candidate_indices]
        _, sorted_indices_of_candidates = torch.sort(candidate_confidences, descending=True)
        top_candidate_indices = sorted_indices_of_candidates[:max_new_labels]
        indices_to_add = candidate_indices[top_candidate_indices]
    else:
        indices_to_add = candidate_indices

    return (indices_to_add, pred), True


def average_model_weights(model_paths, device):
    """Loads multiple model state dicts and averages their weights."""
    avg_state_dict = OrderedDict()
    # Load the first model to initialize the structure
    first_state_dict = torch.load(model_paths[0], map_location=device)
    num_models = len(model_paths)

    for key in first_state_dict.keys():
        avg_state_dict[key] = first_state_dict[key].clone() * (1.0 / num_models)

    # Add the weights from the other models
    for i in range(1, num_models):
        state_dict = torch.load(model_paths[i], map_location=device)
        for key in avg_state_dict.keys():
            avg_state_dict[key] += state_dict[key] * (1.0 / num_models)

    return avg_state_dict


def train_on_df_distributed(job_id, num_jobs, trainer, train_df, iter_dir, lmdb_path):
    """
    Manages the distributed training for one full training phase (multiple epochs).
    """
    print(f"Job {job_id}: Starting distributed training phase.")

    # --- SCATTER (Job 0 Only) ---
    if job_id == 0:
        print("Job 0: Splitting training data for parallel epochs...")
        train_df['char_count'] = train_df['paragraph'].str.len()
        train_df = train_df.sort_values('char_count', ascending=True).drop(columns=['char_count'])
        df_splits = np.array_split(train_df, num_jobs)
        for i, split in enumerate(df_splits):
            split.to_json(os.path.join(iter_dir, f"train_part_{i}.jsonl"), orient="records", lines=True)

    # --- EPOCH LOOP (All Jobs) ---
    for epoch in range(trainer.config['epochs']):
        epoch_dir = os.path.join(iter_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Determine which model to load for this epoch
        if epoch == 0:  # First epoch starts from the model passed into this function
            pass  # trainer.model is already loaded with the correct state
        else:  # Subsequent epochs start from the previously averaged model
            prev_epoch_dir = os.path.join(iter_dir, f"epoch_{epoch}")
            wait_for_file(os.path.join(prev_epoch_dir, "averaging_done.flag"))
            print(f"Job {job_id}: Loading averaged model from epoch {epoch}...")
            trainer.model.load_state_dict(
                torch.load(os.path.join(prev_epoch_dir, "model_averaged.pt"), map_location=trainer.device))

        # All jobs load their data part and train for one epoch
        print(f"Job {job_id}: Starting training for epoch {epoch + 1}...")
        my_train_part_df = robust_jsonl_reader(os.path.join(iter_dir, f"train_part_{job_id}.jsonl"))

        # Setup a fresh optimizer and scaler for each epoch
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=trainer.config['learning_rate'])
        scaler = torch.amp.GradScaler(enabled=trainer.device.type == 'cuda')

        # Create a dataloader for this job's data part
        train_dataset = LMDBParagraphDataset(my_train_part_df, lmdb_path, trainer.tokenizer,
                                             trainer.config['max_length'])
        train_loader = DataLoader(train_dataset, batch_size=trainer.config['batch_size'], shuffle=False,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=4)

        trainer.model.train()
        for batch in tqdm(train_loader, desc=f"Job {job_id} Epoch {epoch + 1}"):
            input_ids, attention_mask, paragraph_sizes, labels = batch
            input_ids, attention_mask, labels = input_ids.to(trainer.device), attention_mask.to(
                trainer.device), labels.to(trainer.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=trainer.device.type == 'cuda'):
                loss = F.cross_entropy(trainer.model(input_ids, attention_mask, paragraph_sizes), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Save this job's model and signal completion for this epoch
        torch.save(trainer.model.state_dict(), os.path.join(epoch_dir, f"model_job{job_id}.pt"))
        with open(os.path.join(epoch_dir, f"epoch_done_job{job_id}.flag"), "w") as f:
            f.write("done")

        # --- AVERAGE (Job 0 Only) ---
        if job_id == 0:
            print(f"Job 0: Waiting for all jobs to finish epoch {epoch + 1}...")
            model_paths = []
            for i in range(num_jobs):
                wait_for_file(os.path.join(epoch_dir, f"epoch_done_job{i}.flag"))
                model_paths.append(os.path.join(epoch_dir, f"model_job{i}.pt"))

            print("Job 0: Averaging model weights...")
            avg_state_dict = average_model_weights(model_paths, trainer.device)
            torch.save(avg_state_dict, os.path.join(epoch_dir, "model_averaged.pt"))

            # Signal that the next epoch can begin
            with open(os.path.join(epoch_dir, "averaging_done.flag"), "w") as f:
                f.write("done")

    # After all epochs, Job 0 loads the final averaged model into the trainer instance
    if job_id == 0:
        final_epoch_dir = os.path.join(iter_dir, f"epoch_{trainer.config['epochs']}")
        print("Job 0: Loading final averaged model into trainer.")
        trainer.model.load_state_dict(
            torch.load(os.path.join(final_epoch_dir, "model_averaged.pt"), map_location=trainer.device))


def train_on_df(trainer, train_df, val_df, lmdb_path):
    print(f"Sorting {len(train_df)} training samples by length for faster epochs...")
    # Ensure the 'paragraph' column exists, which is needed for length calculation
    if 'paragraph' not in train_df.columns:
        raise ValueError("The 'paragraph' column is required in train_df to sort by length.")
    train_df['char_count'] = train_df['paragraph'].str.len()
    train_df = train_df.sort_values('char_count', ascending=True).drop(columns=['char_count'])

    train_loader, val_loader = trainer.create_dataloaders(train_df, val_df, lmdb_path)
    best_f1 = 0
    best_state = trainer.model.state_dict()
    for epoch in range(trainer.config['epochs']):
        trainer.train_epoch(train_loader)
        _, val_f1 = trainer.evaluate(val_loader)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, trainer.model.state_dict()
            print(f"  New best F1: {best_f1:.4f} in epoch {epoch+1}")
    trainer.model.load_state_dict(best_state)
    return best_f1


def wait_for_file(filepath, timeout=54000):  # 10 hour timeout
    """Waits for a file to appear."""
    start_time = time.time()
    while not os.path.exists(filepath):
        time.sleep(30)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Waited too long for file: {filepath}")
    print(f"Found file: {filepath}")


def split_jsonl_file(source_path, target_dir, num_splits):
    """
    Streams a large JSONL file and splits it into multiple smaller files
    without loading the whole file into memory.
    """
    print(f"Memory-efficiently splitting {source_path} into {num_splits} parts...")

    # First, count the total number of lines without loading the file
    with open(source_path, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())

    lines_per_split = math.ceil(total_lines / num_splits)

    output_files = [open(os.path.join(target_dir, f"unlabelled_part_{i}.jsonl"), "w") for i in range(num_splits)]

    current_line = 0
    current_file_idx = 0
    with open(source_path, 'r') as infile:
        for line in tqdm(infile, total=total_lines, desc="Splitting file"):
            if not line.strip():
                continue

            output_files[current_file_idx].write(line)
            current_line += 1

            if current_line >= lines_per_split:
                current_line = 0
                current_file_idx += 1

    for f in output_files:
        f.close()
    print("Splitting complete.")


def predict_logits_on_paragraphs(trainer, df, lmdb_path, batch_size):
    dataset = LMDBParagraphDataset(df, lmdb_path, trainer.tokenizer, trainer.config['max_length'])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    logits_list = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids, attention_mask, p_sizes, _ = batch
            logits = trainer.model(input_ids.to(trainer.device), attention_mask.to(trainer.device), p_sizes)
            logits_list.append(logits.cpu())
    return torch.cat(logits_list, dim=0)


def filter_sort_stream_jsonl(source_path, target_path, char_limit):
    """Memory-efficiently filters and sorts a large JSONL file."""
    print(f"Streaming, filtering (limit: {char_limit} chars), and sorting {source_path}...")
    filtered_lines = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Filtering lines"):
            if not line.strip(): continue
            try:
                record = json.loads(line)
                char_count = len(record.get("paragraph", ""))
                if char_count <= char_limit:
                    filtered_lines.append((line, char_count))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Found {len(filtered_lines)} valid paragraphs within the limit.")
    print("Sorting paragraphs by length...")
    filtered_lines.sort(key=lambda x: x[1])
    print("Writing sorted data...")
    with open(target_path, 'w', encoding='utf-8') as f:
        for line, _ in tqdm(filtered_lines, desc="Writing sorted file"):
            f.write(line)
    print(f"Preprocessing complete. Saved to {target_path}")


def robust_bootstrap_parallel(
        job_id, num_jobs, trainer, train_df, val_df, unlabelled_data_path,
        lmdb_path, output_dir, confidence_threshold, num_iterations, batch_size,
        max_new_labels, min_new_labels, patience
):
    checkpoint_dir = os.path.join(output_dir, "bootstrap_checkpoints")

    # Determine initial state from checkpoint
    start_iter = 1
    start_iter, chk_train, chk_val, chk_unlabelled_path = load_bootstrap_checkpoint(checkpoint_dir, trainer)
    if start_iter is not None:
        train_df, val_df, unlabelled_data_path = chk_train, chk_val, chk_unlabelled_path
        start_iter += 1
    else:
        start_iter = 1

    for it in range(start_iter, num_iterations + 1):
        print(f"\n=== JOB {job_id} | Bootstrapping Iteration {it} ===")
        iter_dir = os.path.join(checkpoint_dir, f"iter_{it}")
        os.makedirs(iter_dir, exist_ok=True)
        best_val_f1 = 0
        patience_counter = 0

        # train_on_df_distributed(job_id, num_jobs, trainer, train_df, iter_dir, lmdb_path)

        # --- TRAINING & SCATTER (Job 0 Only) ---
        if job_id == 0:
            print(f"Job 0: Training model on {len(train_df)} samples...")

            current_val_f1 = train_on_df(trainer, train_df, val_df, lmdb_path)
            torch.save(trainer.model.state_dict(), os.path.join(iter_dir, "model.pt"))

            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                patience_counter = 0
                # torch.save(trainer.model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                print("  Validation F1 improved. Resetting patience counter.")
            else:
                patience_counter += 1
                print(f"  Validation F1 did not improve. Patience counter is now {patience_counter}.")
                if patience_counter >= patience:
                    print(f"  Stopping early due to performance plateau for {patience} iterations.")
                    break

            print("Job 0: Splitting unlabelled data for parallel inference...")
            split_jsonl_file(unlabelled_data_path, iter_dir, num_jobs)

            # Signal to workers that training is done and data is ready
            with open(os.path.join(iter_dir, "inference_ready.flag"), "w") as f: f.write("ready")

        # --- INFERENCE (All Jobs in Parallel) ---
        else:  # Workers wait for the signal
            print(f"Job {job_id}: Waiting for signal from Job 0...")
            wait_for_file(os.path.join(iter_dir, "inference_ready.flag"))
            print(f"Job {job_id}: Signal received. Loading trained model...")
            trainer.model.load_state_dict(torch.load(os.path.join(iter_dir, "model.pt"), map_location=trainer.device))

        # if args.job_id == 0:  # Controller saves the final model and splits inference data
        #     torch.save(trainer.model.state_dict(), os.path.join(iter_dir, "model_final_trained.pt"))
        #     split_jsonl_file(unlabelled_data_path, iter_dir, args.num_jobs)
        #     with open(os.path.join(iter_dir, "inference_ready.flag"), "w") as f:
        #         f.write("ready")
        # else:  # Workers wait
        #     wait_for_file(os.path.join(iter_dir, "inference_ready.flag"))
        #     trainer.model.load_state_dict(
        #         torch.load(os.path.join(iter_dir, "model_final_trained.pt"), map_location=trainer.device))

        print(f"Job {job_id}: Starting inference...")
        part_df = pd.read_json(os.path.join(iter_dir, f"unlabelled_part_{job_id}.jsonl"), lines=True)
        logits = predict_logits_on_paragraphs(trainer, part_df, lmdb_path, batch_size)
        torch.save(logits, os.path.join(iter_dir, f"logits_{job_id}.pt"))

        # Signal that this job's inference is done
        with open(os.path.join(iter_dir, f"inference_job_{job_id}_done.flag"), "w") as f:
            f.write("done")

        # --- GATHER & UPDATE (Job 0 Only) ---
        if job_id == 0:
            print("Job 0: Waiting for all worker jobs to finish inference...")
            for i in range(num_jobs):
                wait_for_file(os.path.join(iter_dir, f"inference_job_{i}_done.flag"))

            print("Job 0: All jobs finished. Gathering results...")
            all_logits = [torch.load(os.path.join(iter_dir, f"logits_{i}.pt")) for i in range(num_jobs)]
            logits = torch.cat(all_logits, dim=0)

            unlabelled_df = pd.read_json(unlabelled_data_path, lines=True)

            info, flag = filter_by_confidence(logits, max_new_labels, min_new_labels, confidence_threshold)
            if not flag:
                break
            indices_to_add, pred = info
            num_new = len(indices_to_add)
            print(f"Job 0: Found {num_new} high-confidence pseudo-labels")
            if num_new == 0:
                print("No more pseudo-labels to add. Stopping.")
                break

            add_df = unlabelled_df.iloc[indices_to_add.cpu().numpy()].copy()
            add_df["label"] = pred[indices_to_add].cpu().numpy()
            train_df = pd.concat([train_df, add_df], ignore_index=True)
            unlabelled_df.drop(indices_to_add.cpu().numpy(), inplace=True)
            unlabelled_df.reset_index(drop=True, inplace=True)
            # train_df = pd.concat([train_df, add_df], ignore_index=True)
            # new_unlabelled_df = unlabelled_df[~mask.cpu().numpy()].reset_index(drop=True)

            next_unlabelled_path = os.path.join(iter_dir, "next_unlabelled.jsonl")
            unlabelled_df.to_json(next_unlabelled_path, orient="records", lines=True)
            unlabelled_data_path = next_unlabelled_path

            save_bootstrap_checkpoint(checkpoint_dir, it, train_df, val_df, unlabelled_df, trainer)

        else:  # Workers are done for this iteration
            print(f"Job {job_id}: Iteration {it} complete. Waiting for next signal.")

    print(f"Job {job_id}: Bootstrapping complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Bootstrapping Trainer")
    parser.add_argument("--job_id", type=int, required=True, help="ID of the current job (e.g., 0, 1).")
    parser.add_argument("--num_jobs", type=int, required=True, help="Total number of parallel jobs.")
    parser.add_argument("--max_new_labels", type=int, default=30000, help="Maximum number of pseudo-labels to add per iteration.")
    parser.add_argument("--min_new_labels", type=int, default=1000, help="Stop if fewer than this many new labels are found.")
    parser.add_argument("--patience", type=int, default=2, help="Stop if validation F1 does not improve for this many iterations.")
    args = parser.parse_args()

    # --- Configuration ---
    LMDB_PATH = "data/tokens_astrobert"
    # MODEL_DIR = "data/bert-classifier-final/output/model_20250621_125159"
    LABELED_DATA_PATH = "data/methodolodgy_labels/labeled_paragraphs.jsonl"
    UNLABELED_DATA_PATH = "data/unlabelled_methodology/unlabelled_paragraphs.jsonl"
    OUTPUT_DIR = "data/bootstrap_output_parallel_v1/"

    custom_config = {
        'learning_rate': 1e-5,
        'batch_size': 4,
        'meta_hidden': 512,
        'dropout': 0.3,
        'epochs': 2,
        'optimizer': 'adam'
    }

    # Initialize trainer
    trainer = BERTMetaClassifierTrainer(
        model_name="adsabs/astroBERT",
        cache_path="/vol/bitbucket/bp824/hf_models/astrobert",
        output_dir=OUTPUT_DIR,
        config=custom_config
    )
    train_df, val_df = trainer.preprocess_data(LABELED_DATA_PATH)
    trainer.load_model_and_tokenizer()
    if args.job_id == 0:
        # unlabelled_df = pd.read_json(UNLABELED_DATA_PATH, lines=True)
        #
        # char_limit = 5000
        # unlabelled_df['char_count'] = unlabelled_df['paragraph'].str.len()
        # unlabelled_df = unlabelled_df[unlabelled_df['char_count'] <= char_limit]
        # unlabelled_df = unlabelled_df.sort_values('char_count', ascending=True).drop(columns=['char_count'])
        # unlabelled_df["label"] = 0
        PREPROCESSED_UNLABELED_PATH = os.path.join(OUTPUT_DIR, "unlabelled_preprocessed.jsonl")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filter_sort_stream_jsonl(
            source_path=UNLABELED_DATA_PATH,
            target_path=PREPROCESSED_UNLABELED_PATH,
            char_limit=8000
        )

        # unlabelled_df.to_json(PREPROCESSED_UNLABELED_PATH, orient="records", lines=True)
        # print(f"Saved preprocessed unlabelled data to {PREPROCESSED_UNLABELED_PATH}")
        # del unlabelled_df
    else:
        PREPROCESSED_UNLABELED_PATH = os.path.join(OUTPUT_DIR, "unlabelled_preprocessed.jsonl")

    inference_batch_size = custom_config.get('batch_size', 4) * 4

    robust_bootstrap_parallel(
        job_id=args.job_id,
        num_jobs=args.num_jobs,
        trainer=trainer,
        train_df=train_df,
        val_df=val_df,
        unlabelled_data_path=PREPROCESSED_UNLABELED_PATH,
        lmdb_path=LMDB_PATH,
        output_dir=OUTPUT_DIR,
        confidence_threshold=0.95,
        num_iterations=5,
        batch_size=inference_batch_size,
        max_new_labels=args.max_new_labels,
        min_new_labels=args.min_new_labels,
        patience=args.patience
    )
