import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from summarizer import Summarizer as BertExtractiveSummarizerRaw
from transformers import AutoModel, AutoTokenizer
import gc
import psutil
import time
from collections import defaultdict
import sqlite3
import pickle
import warnings
import shutil
import math

# Suppress the Pegasus weight initialization warning
warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration were not initialized")

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Model & Tokenizer imports
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer as BertExtractiveSummarizer

# Force output flushing for SLURM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Define the list of models to process
DEFAULT_MODELS = [
    # "bert-extractive-summarizer",
    "facebook/bart-large-cnn",
    "google/pegasus-arxiv",
    # "google/pegasus-cnn_dailymail"
]

class MultiJobSummarizer:
    """Multi-job summarizer with controller-worker pattern for multiple models."""
    
    def __init__(self, args):
        self.args = args
        self.job_id = args.job_id
        self.models_to_process = self.get_models_to_process()
        
        # Base directories
        self.work_dir = Path("data/work_v5")
        self.summaries_base_dir = Path("data/summaries_v5")
        self.splits_dir = self.work_dir / "splits"
        self.flags_dir = self.work_dir / "flags"
        
        # Create base directories
        for dir_path in [self.work_dir, self.summaries_base_dir, self.splits_dir, self.flags_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Memory management
        self.memory_threshold = 0.8
        self.batch_size = args.batch_size
        
        # Global flag files for coordination
        self.data_split_done_flag = self.flags_dir / "data_split_done"
        self.controller_done_flag = self.flags_dir / "controller_done"
        self.worker_done_flag = self.flags_dir / f"worker_{self.job_id}_done"
        self.all_done_flag = self.flags_dir / "all_jobs_done"
        
    def get_models_to_process(self):
        """Get list of models to process."""
        if self.args.models:
            # Use specified models
            return self.args.models
        else:
            # Use default models
            return DEFAULT_MODELS
    
    def get_model_name_safe(self, model_name):
        """Get safe model name for directory creation."""
        return model_name.replace("/", "*")
    
    def setup_progress_db(self, model_name):
        """Setup SQLite database for tracking progress for a specific model."""
        model_name_safe = self.get_model_name_safe(model_name)
        summaries_dir = self.summaries_base_dir / model_name_safe
        job_summaries_dir = summaries_dir / f"job_{self.job_id}"
        job_summaries_dir.mkdir(parents=True, exist_ok=True)
        
        progress_db = job_summaries_dir / "progress.db"
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                status TEXT,
                timestamp REAL,
                error_msg TEXT
            )
        ''')
        conn.commit()
        conn.close()
        return progress_db
    
    def get_processed_papers(self, progress_db):
        """Get set of already processed paper IDs."""
        conn = sqlite3.connect(progress_db)
        cursor = conn.execute("SELECT paper_id FROM processed_papers WHERE status = 'completed'")
        processed = {row[0] for row in cursor.fetchall()}
        conn.close()
        return processed
    
    def mark_paper_processed(self, progress_db, paper_id, status='completed', error_msg=None):
        """Mark a paper as processed in the database."""
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            INSERT OR REPLACE INTO processed_papers (paper_id, status, timestamp, error_msg)
            VALUES (?, ?, ?, ?)
        ''', (paper_id, status, time.time(), error_msg))
        conn.commit()
        conn.close()
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0
    
    def cleanup_memory(self):
        """Force garbage collection and CUDA cache cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def flush_print(self, message):
        """Print with immediate flush for SLURM output."""
        print(f"[Job {self.job_id}] {message}", flush=True)
        sys.stdout.flush()
    
    def wait_for_flag(self, flag_file, timeout=3600):
        """Wait for a flag file to appear."""
        self.flush_print(f"Waiting for flag: {flag_file.name}")
        start_time = time.time()
        while not flag_file.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for {flag_file.name}")
            time.sleep(10)  # Check every 10 seconds
        self.flush_print(f"Flag found: {flag_file.name}")
    
    def create_flag(self, flag_file):
        """Create a flag file."""
        flag_file.touch()
        self.flush_print(f"Created flag: {flag_file.name}")
    
    def split_data(self):
        """Split data into chunks for processing (Controller Job 0 only)."""
        self.flush_print("Starting data splitting...")
        
        # Find all text files
        dataset_dir = Path("data/methodology_dataset")
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        txt_files = list(dataset_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {dataset_dir}")
        
        self.flush_print(f"Found {len(txt_files)} total text files")
        
        # Limit samples if requested
        if self.args.num_samples > 0:
            txt_files = txt_files[:self.args.num_samples]
            self.flush_print(f"Limited to first {self.args.num_samples} files")
        
        # Calculate splits
        total_jobs = self.args.total_jobs
        files_per_job = math.ceil(len(txt_files) / total_jobs)
        
        self.flush_print(f"Splitting {len(txt_files)} files into {total_jobs} jobs ({files_per_job} files per job)")
        
        # Create splits
        for job_id in range(total_jobs):
            start_idx = job_id * files_per_job
            end_idx = min((job_id + 1) * files_per_job, len(txt_files))
            job_files = txt_files[start_idx:end_idx]
            
            if job_files:  # Only create split if there are files
                split_file = self.splits_dir / f"job_{job_id}_files.txt"
                with open(split_file, 'w') as f:
                    for txt_file in job_files:
                        f.write(f"{txt_file}\n")
                self.flush_print(f"Created split for job {job_id}: {len(job_files)} files")
        
        # Create data split done flag
        self.create_flag(self.data_split_done_flag)
        self.flush_print("Data splitting complete")
    
    def load_job_files(self):
        """Load files assigned to this job."""
        split_file = self.splits_dir / f"job_{self.job_id}_files.txt"
        
        if not split_file.exists():
            self.flush_print(f"No split file found for job {self.job_id}")
            return []
        
        files = []
        with open(split_file, 'r') as f:
            for line in f:
                file_path = Path(line.strip())
                if file_path.exists():
                    files.append(file_path)
        
        self.flush_print(f"Loaded {len(files)} files for job {self.job_id}")
        return files
    
    def init_model(self, model_info):
        """Initialize model in a subprocess."""
        model_name, cache_dir, device_preference = model_info
        
        # Set up cache directory
        if cache_dir:
            os.environ['HF_HOME'] = str(cache_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
        
        # Suppress warnings in subprocess
        warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration were not initialized")
        
        summarizer = None
        tokenizer = None
        model = None
        
        try:
            if model_name.lower() == 'bert-extractive-summarizer':
                # model_name = "google-bert/bert-base-uncased"
                if cache_dir:
                    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

                # custom_bert_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
                # custom_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                # summarizer = BertExtractiveSummarizerRaw(custom_model=custom_bert_model, custom_tokenizer=custom_tokenizer)
                    summarizer = BertExtractiveSummarizer()
            else:
                # Use GPU if available and requested
                use_gpu = (device_preference == 0 and torch.cuda.is_available())
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
                
                if use_gpu:
                    model = model.to('cuda')
                    device = 0
                else:
                    device = -1
                
                summarizer = hf_pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    return_tensors=True
                )
            
            return summarizer, tokenizer, model
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            return None, None, None
    
    def generate_summary(self, paper_id, source_text, summarizer, tokenizer, model, model_name):
        """Generate summary for a single paper."""
        summary_data = {
            'paper_id': paper_id,
            'source_text': source_text if self.args.save_source else '',
            'reference_summary': '',
            'generated_summary': '',
            'model_name': model_name,
            'generation_params': {},
            'tokenization_info': {}
        }
        
        if model_name.lower() == 'bert-extractive-summarizer':
            try:
                # Fast pre-check for short inputs
                word_count = len(source_text.strip().split())
                if word_count < 30:
                    raise ValueError("Input too short")

                # Main summarisation
                summary = summarizer(source_text, ratio=0.2)
                if not summary or summary.strip() == "":
                    raise ValueError("Empty summary returned")

                summary_data['generated_summary'] = summary
                summary_data['generation_params'] = {'ratio': 0.2}

            except Exception as e:
                summary_data['generated_summary'] = source_text.strip()
                summary_data['generation_params'] = {'error': str(e), 'fallback': 'source_text_used'}
            
        else:  # Generative models
            max_chunk_tokens = 1024
            chunk_stride = 200

            # Tokenize full input once to split it safely
            inputs = tokenizer(source_text, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"][0]

            if len(input_ids) <= max_chunk_tokens:
                chunks = [source_text]
            else:
                # Chunk text while preserving token boundaries
                chunks = []
                for i in range(0, len(input_ids), max_chunk_tokens - chunk_stride):
                    chunk_ids = input_ids[i:i+max_chunk_tokens]
                    chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    chunks.append(chunk_text)

            # Summarise each chunk individually
            mini_summaries = []
            for chunk in chunks:
                chunk_input = tokenizer(
                    chunk,
                    return_tensors="pt",
                    max_length=max_chunk_tokens,
                    truncation=True,
                    padding=True
                )
                if next(model.parameters()).is_cuda:
                    chunk_input = {k: v.to('cuda') for k, v in chunk_input.items()}
                with torch.no_grad():
                    chunk_output = model.generate(
                        **chunk_input,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                chunk_summary = tokenizer.decode(chunk_output[0], skip_special_tokens=True)
                mini_summaries.append(chunk_summary)

            # Combine summaries and summarise again
            combined_summary_input = " ".join(mini_summaries)
            final_inputs = tokenizer(
                combined_summary_input,
                return_tensors="pt",
                max_length=max_chunk_tokens,
                truncation=True,
                padding=True
            )
            if next(model.parameters()).is_cuda:
                final_inputs = {k: v.to('cuda') for k, v in final_inputs.items()}
            with torch.no_grad():
                final_output = model.generate(
                    **final_inputs,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )
            final_summary = tokenizer.decode(final_output[0], skip_special_tokens=True)

            summary_data['generated_summary'] = final_summary
            summary_data['generation_params'] = {
                'chunked': True,
                'chunk_count': len(chunks),
                'max_length': 150,
                'min_length': 40
            }

            # Save attention weights separately and compressed
            if self.args.save_attention and hasattr(outputs, 'attentions'):
                model_name_safe = self.get_model_name_safe(model_name)
                summaries_dir = self.summaries_base_dir / model_name_safe
                job_summaries_dir = summaries_dir / f"job_{self.job_id}"
                self.save_attention_weights(job_summaries_dir, paper_id, outputs.attentions)
        
        return summary_data
    
    def save_attention_weights(self, job_summaries_dir, paper_id, attention_weights):
        """Save attention weights as compressed pickle files."""
        attention_dir = job_summaries_dir / "attention_weights"
        attention_dir.mkdir(exist_ok=True)
        
        # Convert to numpy and compress
        import numpy as np
        attention_data = []
        for layer_attention in attention_weights:
            attention_data.append([att.cpu().numpy() for att in layer_attention])
        
        # Save as compressed pickle
        with open(attention_dir / f"{paper_id}_attention.pkl.gz", 'wb') as f:
            import gzip
            gzip.GzipFile(fileobj=f).write(pickle.dumps(attention_data))
    
    def save_summary(self, job_summaries_dir, paper_id, summary_data):
        """Save summary data efficiently."""
        # Save JSON (only if needed for analysis)
        if self.args.save_json:
            output_file = job_summaries_dir / f"{paper_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Always save summary text
        summary_txt_file = job_summaries_dir / f"{paper_id}.txt"
        with open(summary_txt_file, "w", encoding="utf-8") as f:
            f.write(summary_data['generated_summary'])
    
    def process_file_batch_for_model(self, file_batch, model_info, model_name):
        """Process a batch of files with a single model instance."""
        # Initialize model for this process
        summarizer, tokenizer, model = self.init_model(model_info)
        
        if summarizer is None:
            self.flush_print(f"Failed to initialize model {model_name}, skipping...")
            return []
        
        # Setup model-specific directories and database
        model_name_safe = self.get_model_name_safe(model_name)
        summaries_dir = self.summaries_base_dir / model_name_safe
        job_summaries_dir = summaries_dir / f"job_{self.job_id}"
        job_summaries_dir.mkdir(parents=True, exist_ok=True)
        
        progress_db = self.setup_progress_db(model_name)
        
        results = []
        
        for txt_file, paper_id in file_batch:
            try:
                # Read document
                with open(txt_file, 'r', encoding='utf-8') as f:
                    document_text = f.read().strip()
                
                if not document_text:
                    self.mark_paper_processed(progress_db, paper_id, 'empty')
                    continue
                
                # Generate summary
                summary_data = self.generate_summary(
                    paper_id, document_text, summarizer, tokenizer, model, model_name
                )
                
                # Save immediately to avoid memory buildup
                self.save_summary(job_summaries_dir, paper_id, summary_data)
                
                # Mark as processed
                self.mark_paper_processed(progress_db, paper_id, 'completed')
                
                results.append(paper_id)
                
            except Exception as e:
                error_msg = f"Error processing {paper_id} with {model_name}: {str(e)}"
                self.mark_paper_processed(progress_db, paper_id, 'error', error_msg)
                print(f"ERROR: {error_msg}")
        
        # Cleanup
        del summarizer, tokenizer, model
        self.cleanup_memory()
        
        return results
    
    def run_summary_generation_for_model(self, txt_files, model_name):
        """Run summary generation for a specific model."""
        if not txt_files:
            self.flush_print(f"No files to process for model {model_name}")
            return
        
        self.flush_print(f"Starting summary generation for {model_name} with {len(txt_files)} files")
        
        # Setup model-specific progress tracking
        progress_db = self.setup_progress_db(model_name)
        
        # Check how many are already processed
        processed = self.get_processed_papers(progress_db)
        remaining_files = [f for f in txt_files if f.stem not in processed]
        
        self.flush_print(f"Model {model_name} - Already processed: {len(processed)}")
        self.flush_print(f"Model {model_name} - Remaining to process: {len(remaining_files)}")
        
        if not remaining_files:
            self.flush_print(f"All files already processed for model {model_name}!")
            return
        
        # Set up cache directory
        if self.args.cache_dir:
            cache_dir = Path(self.args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['HF_HOME'] = str(cache_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
        
        # Process files
        model_info = (model_name, self.args.cache_dir, 
                     0 if torch.cuda.is_available() else -1)
        
        # Create batches
        batches = []
        for i in range(0, len(remaining_files), self.batch_size):
            batch_files = remaining_files[i:i + self.batch_size]
            batch = [(f, f.stem) for f in batch_files]
            batches.append(batch)
        
        self.flush_print(f"Model {model_name} - Created {len(batches)} batches")
        
        # Process batches (single-threaded for better GPU utilization)
        completed_count = len(processed)
        total_files = len(txt_files)
        
        for i, batch in enumerate(batches):
            self.flush_print(f"Model {model_name} - Processing batch {i+1}/{len(batches)}")
            results = self.process_file_batch_for_model(batch, model_info, model_name)
            completed_count += len(results)
            
            # Memory check
            if self.get_memory_usage() > self.memory_threshold:
                self.flush_print(f"Memory usage high ({self.get_memory_usage():.1%}), cleaning up...")
                self.cleanup_memory()
            
            # Progress update
            self.flush_print(f"Model {model_name} - Progress: {completed_count}/{total_files} ({completed_count/total_files:.1%})")
        
        self.flush_print(f"Summary generation complete for {model_name}. Processed {completed_count} files.")
    
    # def run_summary_generation(self, txt_files):
    #     """Run summary generation for all models."""
    #     self.flush_print(f"Starting summary generation for {len(self.models_to_process)} models")
        
    #     for model_idx, model_name in enumerate(self.models_to_process):
    #         self.flush_print(f"Processing model {model_idx + 1}/{len(self.models_to_process)}: {model_name}")
            
    #         try:
    #             self.run_summary_generation_for_model(txt_files, model_name)
    #         except Exception as e:
    #             self.flush_print(f"Error processing model {model_name}: {e}")
    #             continue
            
    #         # Memory cleanup between models
    #         self.cleanup_memory()
            
    #         self.flush_print(f"Completed model {model_name}")
        
    #     self.flush_print("All models processed!")

    def run_summary_generation(self, txt_files):
        """Run summary generation for each batch of files across all models."""
        self.flush_print(f"Starting batched summary generation for {len(txt_files)} files and {len(self.models_to_process)} models")

        # Initialise all models once
        model_infos = {}
        for model_name in self.models_to_process:
            self.flush_print(f"Initialising model: {model_name}")
            model_info = (model_name, self.args.cache_dir, 0 if torch.cuda.is_available() else -1)
            summariser, tokenizer, model = self.init_model(model_info)
            if summariser is None:
                self.flush_print(f"Failed to initialise model {model_name}")
                continue
            model_infos[model_name] = (summariser, tokenizer, model)
        self.flush_print(f"Initialised {len(model_infos)} models")

        # Break files into batches
        batch_size = self.batch_size
        file_batches = [txt_files[i:i + batch_size] for i in range(0, len(txt_files), batch_size)]

        # Loop through each batch of files
        for batch_idx, file_batch in enumerate(file_batches):
            self.flush_print(f"Processing batch {batch_idx + 1}/{len(file_batches)} with {len(file_batch)} files")

            for model_name in self.models_to_process:
                summariser, tokenizer, model = model_infos.get(model_name, (None, None, None))
                if summariser is None:
                    continue

                model_name_safe = self.get_model_name_safe(model_name)
                job_summaries_dir = self.summaries_base_dir / model_name_safe / f"job_{self.job_id}"
                job_summaries_dir.mkdir(parents=True, exist_ok=True)

                for txt_file in file_batch:
                    paper_id = txt_file.stem
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            document_text = f.read().strip()

                        if not document_text:
                            self.flush_print(f"Empty document: {paper_id}")
                            continue

                        summary_data = self.generate_summary(
                            paper_id, document_text, summariser, tokenizer, model, model_name
                        )
                        self.save_summary(job_summaries_dir, paper_id, summary_data)

                    except Exception as e:
                        self.flush_print(f"Error processing {paper_id} with {model_name}: {e}")

                self.cleanup_memory()  # Between model runs to reduce memory growth

        # Cleanup after all batches
        for model_name in model_infos:
            del model_infos[model_name]
        self.cleanup_memory()
        self.flush_print("All batches processed across all models.")

    
    def consolidate_results(self):
        """Consolidate results from all jobs for all models (Controller Job 0 only)."""
        self.flush_print("Consolidating results from all jobs for all models...")
        
        # Wait for all worker jobs to complete
        for job_id in range(1, self.args.total_jobs):
            worker_flag = self.flags_dir / f"worker_{job_id}_done"
            if worker_flag.exists():
                self.flush_print(f"Worker {job_id} completed")
        
        # Consolidate for each model
        for model_name in self.models_to_process:
            model_name_safe = self.get_model_name_safe(model_name)
            model_summaries_dir = self.summaries_base_dir / model_name_safe
            
            self.flush_print(f"Consolidating results for model: {model_name}")
            
            # Copy all summaries to main directory
            total_summaries = 0
            for job_id in range(self.args.total_jobs):
                job_dir = model_summaries_dir / f"job_{job_id}"
                if job_dir.exists():
                    # Copy text files
                    for txt_file in job_dir.glob("*.txt"):
                        dest_file = model_summaries_dir / txt_file.name
                        if not dest_file.exists():
                            shutil.copy2(txt_file, dest_file)
                            total_summaries += 1
                    
                    # Copy JSON files if they exist
                    if self.args.save_json:
                        for json_file in job_dir.glob("*.json"):
                            dest_file = model_summaries_dir / json_file.name
                            if not dest_file.exists():
                                shutil.copy2(json_file, dest_file)
                    
                    # Copy attention weights if they exist
                    attention_dir = job_dir / "attention_weights"
                    if attention_dir.exists():
                        dest_attention_dir = model_summaries_dir / "attention_weights"
                        dest_attention_dir.mkdir(exist_ok=True)
                        for att_file in attention_dir.glob("*.pkl.gz"):
                            dest_file = dest_attention_dir / att_file.name
                            if not dest_file.exists():
                                shutil.copy2(att_file, dest_file)
            
            self.flush_print(f"Consolidated {total_summaries} summaries for model {model_name}")
        
        self.create_flag(self.all_done_flag)
        self.flush_print("All models consolidated!")
    
    def run(self):
        """Main execution method."""
        self.flush_print("=" * 60)
        self.flush_print("STARTING MULTI-MODEL MULTI-JOB SUMMARY GENERATION")
        self.flush_print("=" * 60)
        self.flush_print(f"Models to process: {len(self.models_to_process)}")
        for i, model in enumerate(self.models_to_process):
            self.flush_print(f"  {i+1}. {model}")
        self.flush_print(f"Job ID: {self.job_id}")
        self.flush_print(f"Total Jobs: {self.args.total_jobs}")
        
        if self.job_id == 0:
            # Controller job (Job 0)
            self.flush_print("Running as CONTROLLER job")
            
            # Step 1: Split data
            if not self.data_split_done_flag.exists():
                self.split_data()
            else:
                self.flush_print("Data already split")
            
            # Step 2: Process own files with all models
            txt_files = self.load_job_files()
            self.run_summary_generation(txt_files)
            
            # Step 3: Mark controller done
            self.create_flag(self.controller_done_flag)
            
            # Step 4: Wait for all workers and consolidate
            self.consolidate_results()
            
        else:
            # Worker job (Job 1+)
            self.flush_print("Running as WORKER job")
            
            # Step 1: Wait for data split
            self.wait_for_flag(self.data_split_done_flag)
            
            # Step 2: Process assigned files with all models
            txt_files = self.load_job_files()
            self.run_summary_generation(txt_files)
            
            # Step 3: Mark worker done
            self.create_flag(self.worker_done_flag)
        
        self.flush_print("=" * 60)
        self.flush_print("JOB COMPLETE")
        self.flush_print("=" * 60)


def main(args):
    """Main function using the multi-job summarizer."""
    summarizer = MultiJobSummarizer(args)
    summarizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Job Summary Generation (Controller-Worker Pattern)")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                       help="List of model identifiers to process. If not specified, uses default model list.")
    parser.add_argument("--job_id", type=int, required=True,
                       help="Job ID (0 = controller, 1+ = worker).")
    parser.add_argument("--total_jobs", type=int, required=True,
                       help="Total number of jobs.")
    parser.add_argument("--num_samples", type=int, default=0,
                       help="Number of samples to process (0 = all files).")
    parser.add_argument("--batch_size", type=int, default=150,
                       help="Number of papers to process in each batch.")
    parser.add_argument("--save_attention", action="store_true",
                       help="Save attention weights (WARNING: requires massive disk space).")
    parser.add_argument("--save_tokenization_info", action="store_true",
                       help="Save detailed tokenization information.")
    parser.add_argument("--save_json", action="store_true",
                       help="Save full JSON metadata (uses more disk space).")
    parser.add_argument("--save_source", action="store_true",
                       help="Save source text in JSON (uses much more disk space).")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache HuggingFace models and tokenizers.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.job_id < 0 or args.job_id >= args.total_jobs:
        raise ValueError(f"Job ID {args.job_id} must be between 0 and {args.total_jobs - 1}")
    
    if args.save_attention:
        print("WARNING: Saving attention weights for multiple models will require massive disk space!")
        if args.job_id == 0:  # Only ask controller
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                exit()
    
    main(args)