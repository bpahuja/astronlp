import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import re
import math
import time
import numpy as np
from bootstrap_multi import (
    BERTMetaClassifier,
    LMDBParagraphDataset,
    collate_fn,
    single_sample_collate_fn,
    predict_logits_on_paragraphs,
    BERTMetaClassifierTrainer,
    robust_jsonl_reader
)


def clean_math_tags(paragraph):
    return re.sub(r'\[MATH_tex=.*?\]', '[MATH]', paragraph)


def wait_for_file(filepath, timeout=36000):  # 1 hour timeout for inference
    """Waits for a file to appear."""
    start_time = time.time()
    while not os.path.exists(filepath):
        time.sleep(5)  # Check every 5 seconds for faster response
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Waited too long for file: {filepath}")
    print(f"Found file: {filepath}")


def split_jsonl_file_for_inference(source_path, target_dir, num_splits, job_id=None):
    """
    Streams a large JSONL file and splits it into multiple smaller files
    without loading the whole file into memory.
    If job_id is provided, only creates that specific split.
    """
    if job_id is not None:
        print(f"Creating split {job_id} from {source_path}...")
    else:
        print(f"Memory-efficiently splitting {source_path} into {num_splits} parts...")

    # First, count the total number of lines without loading the file
    with open(source_path, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())

    lines_per_split = math.ceil(total_lines / num_splits)

    if job_id is not None:
        # Only create the specific split for this job
        start_line = job_id * lines_per_split
        end_line = min((job_id + 1) * lines_per_split, total_lines)

        output_file = open(os.path.join(target_dir, f"input_part_{job_id}.jsonl"), "w")

        current_line = 0
        with open(source_path, 'r') as infile:
            for line in infile:
                if not line.strip():
                    continue
                if start_line <= current_line < end_line:
                    output_file.write(line)
                current_line += 1
                if current_line >= end_line:
                    break

        output_file.close()
        print(f"Created split {job_id} with {end_line - start_line} lines")
    else:
        # Create all splits (original behavior)
        output_files = [open(os.path.join(target_dir, f"input_part_{i}.jsonl"), "w") for i in range(num_splits)]

        current_line = 0
        current_file_idx = 0
        with open(source_path, 'r') as infile:
            for line in tqdm(infile, total=total_lines, desc="Splitting file"):
                if not line.strip():
                    continue

                output_files[current_file_idx].write(line)
                current_line += 1

                if current_line >= lines_per_split and current_file_idx < num_splits - 1:
                    current_line = 0
                    current_file_idx += 1

        for f in output_files:
            f.close()
        print("Splitting complete.")


def merge_prediction_files(work_dir, num_jobs, output_file):
    """
    Merges prediction files from all jobs into a single output file.
    """
    print(f"Merging {num_jobs} prediction files into {output_file}...")

    with open(output_file, 'w') as outfile:
        for job_id in range(num_jobs):
            part_file = os.path.join(work_dir, f"predictions_part_{job_id}.jsonl")
            if os.path.exists(part_file):
                with open(part_file, 'r') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
            else:
                print(f"Warning: Missing prediction file for job {job_id}")

    print(f"Merged predictions saved to {output_file}")


def run_distributed_inference(args):
    """
    Main function to orchestrate the distributed inference process.
    """
    # --- Setup ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires at least one GPU.")
        return

    # Create working directory for this inference run
    work_dir = os.path.join(os.path.dirname(args.output_file), "inference_work")
    os.makedirs(work_dir, exist_ok=True)

    LMDB_PATH = "data/tokens_astrobert"
    custom_config = {
        'learning_rate': 1e-5,
        'batch_size': args.batch_size,
        'meta_hidden': args.meta_hidden,
        'dropout': args.dropout,
        'epochs': 2,
        'optimizer': 'adam',
        'num_classes': args.num_classes
    }

    # Initialize trainer
    trainer = BERTMetaClassifierTrainer(
        model_name=args.model_name,
        cache_path=args.cache_path,
        output_dir=work_dir,
        config=custom_config
    )
    trainer.load_model_and_tokenizer()

    print(f"Loading model weights from: {args.model_pt_path}")
    if not os.path.exists(args.model_pt_path):
        raise FileNotFoundError(f"Model weights file not found: {args.model_pt_path}")
    trainer.model.load_state_dict(torch.load(args.model_pt_path, map_location=trainer.device))

    # --- JOB COORDINATION ---

    if args.job_id == 0:
        # Job 0: Split the input file and coordinate
        print("Job 0: Splitting input file for parallel processing...")
        split_jsonl_file_for_inference(args.input_file, work_dir, args.num_jobs)

        # Signal that splitting is complete
        with open(os.path.join(work_dir, "splitting_done.flag"), "w") as f:
            f.write("done")
        print("Job 0: Input file splitting complete.")
    else:
        # Other jobs: Wait for splitting to complete
        print(f"Job {args.job_id}: Waiting for input file splitting...")
        wait_for_file(os.path.join(work_dir, "splitting_done.flag"))
        print(f"Job {args.job_id}: Splitting complete, proceeding with inference.")

    # --- PARALLEL INFERENCE ---

    # Each job processes its assigned part
    part_file = os.path.join(work_dir, f"input_part_{args.job_id}.jsonl")

    if not os.path.exists(part_file):
        print(f"Job {args.job_id}: No input part found, creating empty output.")
        # Create empty output file for this job
        with open(os.path.join(work_dir, f"predictions_part_{args.job_id}.jsonl"), "w") as f:
            pass
    else:
        print(f"Job {args.job_id}: Processing {part_file}...")

        # Load and process the data part
        try:
            unlabelled_df = robust_jsonl_reader(part_file)
            print(f"Job {args.job_id}: Loaded {len(unlabelled_df)} samples for inference.")

            # Run inference on this part
            logits = predict_logits_on_paragraphs(trainer, unlabelled_df, LMDB_PATH, args.batch_size)
            probs = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)
            confidence_scores = torch.max(probs, dim=1)[0]

            # Add predictions to dataframe
            unlabelled_df['predicted_label'] = predicted_labels.cpu().numpy()
            unlabelled_df['confidence'] = confidence_scores.cpu().numpy()

            # Save predictions for this part
            part_output = os.path.join(work_dir, f"predictions_part_{args.job_id}.jsonl")
            unlabelled_df.to_json(part_output, orient="records", lines=True)
            print(f"Job {args.job_id}: Saved predictions to {part_output}")

        except Exception as e:
            print(f"Job {args.job_id}: Error processing data: {e}")
            # Create empty output file to prevent hanging
            with open(os.path.join(work_dir, f"predictions_part_{args.job_id}.jsonl"), "w") as f:
                pass

    # Signal that this job's inference is complete
    with open(os.path.join(work_dir, f"inference_job_{args.job_id}_done.flag"), "w") as f:
        f.write("done")

    # --- MERGING RESULTS (Job 0 only) ---

    if args.job_id == 0:
        print("Job 0: Waiting for all jobs to complete inference...")

        # Wait for all jobs to finish
        for i in range(args.num_jobs):
            wait_for_file(os.path.join(work_dir, f"inference_job_{i}_done.flag"))

        print("Job 0: All jobs completed. Merging results...")
        merge_prediction_files(work_dir, args.num_jobs, args.output_file)

        # Optional: Clean up intermediate files
        if not args.keep_intermediate:
            print("Job 0: Cleaning up intermediate files...")
            for i in range(args.num_jobs):
                for filename in [f"input_part_{i}.jsonl", f"predictions_part_{i}.jsonl",
                                 f"inference_job_{i}_done.flag"]:
                    filepath = os.path.join(work_dir, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
            # Remove flags
            for flag_file in ["splitting_done.flag"]:
                filepath = os.path.join(work_dir, flag_file)
                if os.path.exists(filepath):
                    os.remove(filepath)

        print("Job 0: Distributed inference complete!")
    else:
        print(f"Job {args.job_id}: Inference complete. Waiting for Job 0 to merge results.")


def run_single_inference(args):
    """
    Original single-job inference function for compatibility.
    """
    # --- Setup ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires at least one GPU.")
        return

    LMDB_PATH = "data/tokens_astrobert"
    custom_config = {
        'learning_rate': 1e-5,
        'batch_size': args.batch_size,
        'meta_hidden': args.meta_hidden,
        'dropout': args.dropout,
        'epochs': 2,
        'optimizer': 'adam',
        'num_classes': args.num_classes
    }

    # Initialize trainer
    trainer = BERTMetaClassifierTrainer(
        model_name=args.model_name,
        cache_path=args.cache_path,
        output_dir="data/inference",
        config=custom_config
    )
    trainer.load_model_and_tokenizer()

    print(f"Loading model weights from: {args.model_pt_path}")
    if not os.path.exists(args.model_pt_path):
        raise FileNotFoundError(f"Model weights file not found: {args.model_pt_path}")
    trainer.model.load_state_dict(torch.load(args.model_pt_path, map_location=trainer.device))

    unlabelled_df = pd.read_json(args.input_file, lines=True)

    logits = predict_logits_on_paragraphs(trainer, unlabelled_df, LMDB_PATH, args.batch_size)
    probs = torch.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probs, dim=1)
    confidence_scores = torch.max(probs, dim=1)[0]

    unlabelled_df['predicted_label'] = predicted_labels.cpu().numpy()
    unlabelled_df['confidence'] = confidence_scores.cpu().numpy()
    unlabelled_df.to_json(args.output_file, orient="records", lines=True)

    print("Single job inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-performance, parallel inference script for PyTorch models.")

    # --- DISTRIBUTED JOB ARGUMENTS ---
    parser.add_argument("--job_id", type=int, default=None,
                        help="ID of the current job (e.g., 0, 1, 2). If not provided, runs single-job inference.")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Total number of parallel jobs. Only used if job_id is provided.")

    # --- MODEL ARGUMENTS ---
    parser.add_argument("--model_pt_path", type=str, required=True,
                        help="Direct path to the trained model's .pt (state_dict) file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face name of the base model (e.g., 'adsabs/astroBERT').")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Optional path to cache Hugging Face models.")
    parser.add_argument("--meta_hidden", type=int, default=512,
                        help="Size of the LSTM hidden layer.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate used in the model.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes for the model.")

    # --- DATA AND PERFORMANCE ARGUMENTS ---
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input .jsonl file with data for inference.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the output .jsonl file with predictions.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="The batch size per job.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of CPU workers for the DataLoader.")
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="Keep intermediate files after completion (useful for debugging).")

    args = parser.parse_args()

    # Determine whether to run distributed or single inference
    if args.job_id is not None:
        if args.num_jobs <= 1:
            print("Warning: num_jobs should be > 1 for distributed inference. Running single job.")
            run_single_inference(args)
        else:
            run_distributed_inference(args)
    else:
        run_single_inference(args)