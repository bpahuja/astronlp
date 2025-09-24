import os
import sys
import argparse
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil
import time
import sqlite3
import warnings
import shutil
import math
import threading
import queue
import signal
import atexit
from typing import List, Tuple, Dict, Any, Optional
import random
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Force output flushing for SLURM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# SPECTER 2 model
SPECTER2_MODEL = "sentence-transformers/allenai-specter"

class MultiJobParagraphEmbedder:
    """Multi-job paragraph embedder with controller-worker pattern for SPECTER 2."""
    
    def __init__(self, args):
        self.args = args
        self.job_id = args.job_id
        self.model_name = SPECTER2_MODEL  # Fixed to SPECTER 2
        self.root = Path("/vol/bitbucket/bp824/astro")
        
        # Base directories
        self.work_dir = self.root / "data/paragraph_embeddings_work_v4"
        self.embeddings_base_dir =  self.root / "data/paragraph_embeddings_specter2_v4"
        self.splits_dir = self.work_dir / "splits"
        self.flags_dir = self.work_dir / "flags"
        
        # Create base directories
        for dir_path in [self.work_dir, self.embeddings_base_dir, self.splits_dir, self.flags_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Memory management
        self.memory_threshold = 0.8
        self.batch_size = args.batch_size
        
        # GPU setup
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device_queue = queue.Queue()
        for i in range(self.num_gpus):
            self.device_queue.put(i)
        
        # Model instance (load once)
        self.model = None
        
        # Global flag files for coordination
        self.data_split_done_flag = self.flags_dir / "data_split_done"
        self.controller_done_flag = self.flags_dir / "controller_done"
        self.worker_done_flag = self.flags_dir / f"worker_{self.job_id}_done"
        self.all_done_flag = self.flags_dir / "all_jobs_done"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        atexit.register(self.cleanup_on_exit)
        
        self.shutdown_event = threading.Event()
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.flush_print(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def cleanup_on_exit(self):
        """Cleanup on exit."""
        self.flush_print("Cleaning up on exit...")
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_name_safe(self, model_name):
        """Get safe model name for directory creation."""
        return model_name.replace("/", "*")
    
    def split_paragraphs(self, text):
        """Split text into paragraphs and return non-empty paragraphs."""
        # Split by double newlines first, then single newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If no double newlines, split by single newlines
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            # Filter out very short paragraphs (less than 10 characters)
            if len(para) > 10:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def generate_paragraph_id(self, paper_id, para_index):
        """Generate unique paragraph ID."""
        return f"{paper_id}_para_{para_index:04d}"
    
    def setup_progress_db(self):
        """Setup SQLite database for tracking progress for paragraph embeddings."""
        embeddings_dir = self.embeddings_base_dir
        job_embeddings_dir = embeddings_dir / f"job_{self.job_id}"
        job_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        progress_db = job_embeddings_dir / "progress.db"
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                status TEXT,
                timestamp REAL,
                error_msg TEXT,
                num_paragraphs INTEGER
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_paragraphs (
                paragraph_id TEXT PRIMARY KEY,
                paper_id TEXT,
                paragraph_index INTEGER,
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
    
    def mark_paper_processed(self, progress_db, paper_id, status='completed', error_msg=None, num_paragraphs=0):
        """Mark a paper as processed in the database."""
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            INSERT OR REPLACE INTO processed_papers (paper_id, status, timestamp, error_msg, num_paragraphs)
            VALUES (?, ?, ?, ?, ?)
        ''', (paper_id, status, time.time(), error_msg, num_paragraphs))
        conn.commit()
        conn.close()
    
    def mark_paragraph_processed(self, progress_db, paragraph_id, paper_id, para_index, status='completed', error_msg=None):
        """Mark a paragraph as processed in the database."""
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            INSERT OR REPLACE INTO processed_paragraphs (paragraph_id, paper_id, paragraph_index, status, timestamp, error_msg)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (paragraph_id, paper_id, para_index, status, time.time(), error_msg))
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
        
        # Use fixed input directory
        input_dir = Path(self.args.input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {input_dir}")
        
        self.flush_print(f"Found {len(txt_files)} total text files in {input_dir}")
        
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
    
    def init_model(self):
        """Initialize SPECTER 2 model once."""
        if self.model is not None:
            return self.model
            
        # Set up cache directory
        if self.args.cache_dir:
            cache_dir = Path(self.args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
        
        try:
            # Initialize model
            if torch.cuda.is_available() and self.num_gpus > 0:
                device = f'cuda:0'
                torch.cuda.set_device(0)
                self.flush_print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                self.flush_print("Using CPU (CUDA not available)")
            
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=self.args.cache_dir, 
                device=device
            )
            
            self.flush_print(f"SPECTER 2 model loaded on device {device}")
            return self.model
                
        except Exception as e:
            self.flush_print(f"Error initializing SPECTER 2 model: {e}")
            return None
    
    def generate_paragraph_embeddings_batch(self, paragraphs_data: List[Dict]):
        """Generate embeddings for a batch of paragraphs using the pre-loaded model."""
        if self.model is None:
            self.flush_print("Model not initialized!")
            return []
            
        try:
            # Extract texts
            texts = [item['text'] for item in paragraphs_data]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=min(self.batch_size, len(texts)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Package results
            results = []
            for i, (para_data, embedding) in enumerate(zip(paragraphs_data, embeddings)):
                results.append({
                    'paragraph_id': para_data['paragraph_id'],
                    'paper_id': para_data['paper_id'],
                    'paragraph_index': para_data['paragraph_index'],
                    'embedding': embedding,
                    'text_length': len(para_data['text']),
                    'embedding_dim': embedding.shape[0]
                })
            
            return results
            
        except Exception as e:
            self.flush_print(f"Error generating paragraph embeddings: {e}")
            return []
    
    def save_paragraph_embeddings_batch(self, job_embeddings_dir: Path, batch_results: List[Dict], batch_idx: int):
        """Save batch of paragraph embeddings."""
        batch_file = job_embeddings_dir / f"paragraph_batch_{batch_idx:06d}.npz"
        
        # Extract embeddings and metadata
        embeddings = np.array([result['embedding'] for result in batch_results])
        paragraph_ids = [result['paragraph_id'] for result in batch_results]
        paper_ids = [result['paper_id'] for result in batch_results]
        paragraph_indices = [result['paragraph_index'] for result in batch_results]
        text_lengths = [result['text_length'] for result in batch_results]
        
        # Save as compressed numpy file
        np.savez_compressed(
            batch_file,
            embeddings=embeddings,
            paragraph_ids=paragraph_ids,
            paper_ids=paper_ids,
            paragraph_indices=paragraph_indices,
            text_lengths=text_lengths,
            model_name=self.model_name,
            embedding_dim=batch_results[0]['embedding_dim'] if batch_results else 0
        )
        
        return len(batch_results)
    
    def process_file_batch(self, file_batch, batch_idx, progress_db, job_embeddings_dir):
        """Process a batch of files and extract paragraphs (model already loaded)."""
        # Extract paragraphs from files
        paragraphs_data = []
        
        for txt_file, paper_id in file_batch:
            try:
                # Read document
                with open(txt_file, 'r', encoding='utf-8') as f:
                    document_text = f.read().strip()
                
                if document_text:
                    # Split into paragraphs
                    paragraphs = self.split_paragraphs(document_text)
                    
                    if paragraphs:
                        for para_idx, paragraph in enumerate(paragraphs):
                            paragraph_id = self.generate_paragraph_id(paper_id, para_idx)
                            paragraphs_data.append({
                                'paragraph_id': paragraph_id,
                                'paper_id': paper_id,
                                'paragraph_index': para_idx,
                                'text': paragraph
                            })
                        
                        self.mark_paper_processed(progress_db, paper_id, 'completed', num_paragraphs=len(paragraphs))
                    else:
                        self.mark_paper_processed(progress_db, paper_id, 'no_paragraphs')
                else:
                    self.mark_paper_processed(progress_db, paper_id, 'empty')
                    
            except Exception as e:
                error_msg = f"Error reading {paper_id}: {str(e)}"
                self.mark_paper_processed(progress_db, paper_id, 'error', error_msg)
                self.flush_print(f"ERROR: {error_msg}")
        
        results = []
        
        if paragraphs_data:
            # Generate embeddings for all paragraphs using pre-loaded model
            batch_results = self.generate_paragraph_embeddings_batch(paragraphs_data)
            
            if batch_results:
                # Save embeddings batch
                saved_count = self.save_paragraph_embeddings_batch(job_embeddings_dir, batch_results, batch_idx)
                
                # Mark paragraphs as completed
                for result in batch_results:
                    self.mark_paragraph_processed(
                        progress_db, 
                        result['paragraph_id'], 
                        result['paper_id'], 
                        result['paragraph_index'], 
                        'completed'
                    )
                
                results = [result['paragraph_id'] for result in batch_results]
                self.flush_print(f"Processed {len(paragraphs_data)} paragraphs from {len(file_batch)} papers")
        
        return results
    
    def run_paragraph_embedding_generation(self, txt_files):
        """Run paragraph embedding generation with model loaded once."""
        if not txt_files:
            self.flush_print("No files to process")
            return
        
        self.flush_print(f"Starting paragraph embedding generation with SPECTER 2 for {len(txt_files)} files")
        
        # Setup progress tracking
        progress_db = self.setup_progress_db()
        job_embeddings_dir = self.embeddings_base_dir / f"job_{self.job_id}"
        job_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Check how many are already processed
        processed = self.get_processed_papers(progress_db)
        remaining_files = [f for f in txt_files if f.stem not in processed]
        
        self.flush_print(f"Already processed: {len(processed)} papers")
        self.flush_print(f"Remaining to process: {len(remaining_files)} papers")
        
        if not remaining_files:
            self.flush_print("All files already processed!")
            return
        
        # Initialize model ONCE
        self.flush_print("Initializing SPECTER 2 model...")
        if self.init_model() is None:
            self.flush_print("Failed to initialize model, exiting...")
            return
        
        # Create batches of files (not paragraphs)
        file_batch_size = max(1, self.batch_size // 10)  # Smaller batches since we're processing multiple paragraphs per file
        batches = []
        for i in range(0, len(remaining_files), file_batch_size):
            batch_files = remaining_files[i:i + file_batch_size]
            batch = [(f, f.stem) for f in batch_files]
            batches.append(batch)
        
        self.flush_print(f"Created {len(batches)} file batches (batch size: {file_batch_size})")
        
        # Process batches using the same model instance
        completed_files = len(processed)
        total_files = len(txt_files)
        total_paragraphs = 0
        
        # Track timing for ETA calculation
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(batches):
            if self.shutdown_event.is_set():
                break
            
            batch_start_time = time.time()
            self.flush_print(f"Processing file batch {i+1}/{len(batches)} ({len(batch)} files)")
            
            # Process batch using pre-loaded model
            results = self.process_file_batch(batch, i, progress_db, job_embeddings_dir)
            completed_files += len(batch)
            total_paragraphs += len(results)
            
            # Track batch processing time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Calculate ETA
            batches_completed = i + 1
            batches_remaining = len(batches) - batches_completed
            
            if batches_completed >= 3:  # Wait for at least 3 batches for stable estimate
                # Use average of recent batch times (last 5 batches or all if less than 5)
                recent_times = batch_times[-5:]
                avg_batch_time = sum(recent_times) / len(recent_times)
                eta_seconds = avg_batch_time * batches_remaining
                
                # Format ETA
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                
                # Calculate processing rate
                elapsed_time = time.time() - start_time
                files_per_minute = (completed_files - len(processed)) / (elapsed_time / 60) if elapsed_time > 0 else 0
                paragraphs_per_minute = total_paragraphs / (elapsed_time / 60) if elapsed_time > 0 else 0
                
                self.flush_print(f"Batch {batches_completed}/{len(batches)} done in {batch_time:.1f}s | "
                               f"ETA: {eta_str} | Rate: {files_per_minute:.1f} files/min, {paragraphs_per_minute:.1f} paras/min")
            else:
                self.flush_print(f"Batch {batches_completed}/{len(batches)} done in {batch_time:.1f}s | "
                               f"Calculating ETA... (need {3-batches_completed} more batches)")
            
            # Memory check and cleanup (but keep model loaded)
            if self.get_memory_usage() > self.memory_threshold:
                self.flush_print(f"Memory usage high ({self.get_memory_usage():.1%}), cleaning up...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Progress update
            progress_pct = completed_files / total_files * 100
            self.flush_print(f"Overall Progress: {completed_files}/{total_files} files ({progress_pct:.1f}%), {total_paragraphs} total paragraphs")
        
        # Final summary with total time
        total_time = time.time() - start_time
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}m"
        else:
            time_str = f"{total_time/3600:.1f}h"
        
        self.flush_print(f"Paragraph embedding generation complete in {time_str}.")
        self.flush_print(f"Processed {completed_files} files, {total_paragraphs} paragraphs.")
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            self.flush_print(f"Average batch time: {avg_batch_time:.1f}s")
        self.flush_print("Model will be cleaned up at job completion.")
    
    def consolidate_paragraph_embeddings(self):
        """Consolidate paragraph embeddings from all job directories."""
        self.flush_print("Consolidating paragraph embeddings from all jobs...")
        
        # Collect all embeddings and metadata from all jobs
        all_embeddings = []
        all_paragraph_ids = []
        all_paper_ids = []
        all_paragraph_indices = []
        all_text_lengths = []
        
        for job_id in range(self.args.total_jobs):
            job_dir = self.embeddings_base_dir / f"job_{job_id}"
            if job_dir.exists():
                # Load all batch files for this job
                for batch_file in sorted(job_dir.glob("paragraph_batch_*.npz")):
                    try:
                        data = np.load(batch_file, allow_pickle=True)
                        embeddings = data['embeddings']
                        paragraph_ids = data['paragraph_ids'].tolist()
                        paper_ids = data['paper_ids'].tolist()
                        paragraph_indices = data['paragraph_indices'].tolist()
                        text_lengths = data['text_lengths'].tolist()
                        
                        all_embeddings.extend(embeddings)
                        all_paragraph_ids.extend(paragraph_ids)
                        all_paper_ids.extend(paper_ids)
                        all_paragraph_indices.extend(paragraph_indices)
                        all_text_lengths.extend(text_lengths)
                        
                    except Exception as e:
                        self.flush_print(f"Error loading batch file {batch_file}: {e}")
        
        if not all_embeddings:
            self.flush_print("No paragraph embeddings found")
            return
        
        # Convert to numpy array
        embeddings_matrix = np.array(all_embeddings)
        
        # Save consolidated embeddings
        embed_save_path = self.embeddings_base_dir / "paragraph_embeddings.npy"
        map_save_path = self.embeddings_base_dir / "paragraph_id_to_idx.json"
        paper_map_save_path = self.embeddings_base_dir / "paragraph_to_paper_mapping.json"
        
        np.save(embed_save_path, embeddings_matrix)
        
        # Create paragraph ID to index mapping
        paragraphid_to_idx = {pid: idx for idx, pid in enumerate(all_paragraph_ids)}
        with open(map_save_path, 'w') as f:
            json.dump(paragraphid_to_idx, f, indent=2)
        
        # Create paragraph to paper mapping
        paragraph_to_paper = {}
        for i, (para_id, paper_id, para_idx, text_len) in enumerate(zip(
            all_paragraph_ids, all_paper_ids, all_paragraph_indices, all_text_lengths)):
            paragraph_to_paper[para_id] = {
                'paper_id': paper_id,
                'paragraph_index': para_idx,
                'text_length': text_len,
                'embedding_index': i
            }
        
        with open(paper_map_save_path, 'w') as f:
            json.dump(paragraph_to_paper, f, indent=2)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_paragraphs': len(all_paragraph_ids),
            'num_unique_papers': len(set(all_paper_ids)),
            'embedding_dim': embeddings_matrix.shape[1] if embeddings_matrix.size > 0 else 0,
            'embedding_shape': list(embeddings_matrix.shape),
            'avg_text_length': np.mean(all_text_lengths) if all_text_lengths else 0,
            'timestamp': time.time()
        }
        
        metadata_path = self.embeddings_base_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary statistics
        paper_paragraph_counts = {}
        for paper_id in all_paper_ids:
            paper_paragraph_counts[paper_id] = paper_paragraph_counts.get(paper_id, 0) + 1
        
        stats = {
            'total_paragraphs': len(all_paragraph_ids),
            'total_papers': len(paper_paragraph_counts),
            'avg_paragraphs_per_paper': np.mean(list(paper_paragraph_counts.values())),
            'max_paragraphs_per_paper': max(paper_paragraph_counts.values()),
            'min_paragraphs_per_paper': min(paper_paragraph_counts.values())
        }
        
        stats_path = self.embeddings_base_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.flush_print("Consolidated paragraph embeddings:")
        self.flush_print(f"  Shape: {embeddings_matrix.shape}")
        self.flush_print(f"  Total paragraphs: {len(all_paragraph_ids)}")
        self.flush_print(f"  Unique papers: {len(set(all_paper_ids))}")
        self.flush_print(f"  Avg paragraphs per paper: {stats['avg_paragraphs_per_paper']:.1f}")
        self.flush_print(f"  Saved to: {embed_save_path}")
        self.flush_print(f"  Paragraph mapping: {map_save_path}")
        self.flush_print(f"  Paper mapping: {paper_map_save_path}")
    
    def consolidate_results(self):
        """Consolidate results from all jobs (Controller Job 0 only)."""
        self.flush_print("Consolidating results from all jobs...")
        
        # Wait for all worker jobs to complete
        incomplete_workers = list(range(1, self.args.total_jobs))
        
        while incomplete_workers:
            still_waiting = []
            for job_id in incomplete_workers:
                worker_flag = self.flags_dir / f"worker_{job_id}_done"
                if worker_flag.exists():
                    self.flush_print(f"Worker {job_id} completed")
                else:
                    still_waiting.append(job_id)
            
            incomplete_workers = still_waiting
            
            if incomplete_workers:
                self.flush_print(f"Still waiting for workers: {incomplete_workers}")
                time.sleep(10)  # Wait 10 seconds before checking again
        
        self.flush_print("All workers completed!")
        
        # Consolidate paragraph embeddings
        self.consolidate_paragraph_embeddings()
        
        self.create_flag(self.all_done_flag)
        self.flush_print("All paragraph embeddings consolidated!")
    
    def run(self):
        """Main execution method."""
        self.flush_print("=" * 60)
        self.flush_print("STARTING PARAGRAPH-LEVEL EMBEDDING GENERATION WITH SPECTER 2")
        self.flush_print("=" * 60)
        self.flush_print(f"Input directory: {self.args.input_dir}")
        self.flush_print(f"Embedding model: {self.model_name}")
        self.flush_print(f"Job ID: {self.job_id}")
        self.flush_print(f"Total Jobs: {self.args.total_jobs}")
        self.flush_print(f"Available GPUs: {self.num_gpus}")
        
        if self.job_id == 0:
            # Controller job (Job 0)
            self.flush_print("Running as CONTROLLER job")
            
            # Step 1: Split data
            if not self.data_split_done_flag.exists():
                self.split_data()
            else:
                self.flush_print("Data already split")
            
            # Step 2: Process own files
            txt_files = self.load_job_files()
            self.run_paragraph_embedding_generation(txt_files)
            
            # Step 3: Mark controller done
            self.create_flag(self.controller_done_flag)
            
            # Step 4: Wait for all workers and consolidate
            self.consolidate_results()
            
        else:
            # Worker job (Job 1+)
            self.flush_print("Running as WORKER job")
            
            # Step 1: Wait for data split
            self.wait_for_flag(self.data_split_done_flag)
            
            # Step 2: Process assigned files
            txt_files = self.load_job_files()
            self.run_paragraph_embedding_generation(txt_files)
            
            # Step 3: Mark worker done
            self.create_flag(self.worker_done_flag)
        
        self.flush_print("=" * 60)
        self.flush_print("JOB COMPLETE")
        self.flush_print("=" * 60)


def main(args):
    """Main function using the multi-job paragraph embedder."""
    embedder = MultiJobParagraphEmbedder(args)
    embedder.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paragraph-Level Embedding Generation with SPECTER 2 (Controller-Worker Pattern)")
    parser.add_argument("--job_id", type=int, required=True,
                       help="Job ID (0 = controller, 1+ = worker).")
    parser.add_argument("--total_jobs", type=int, required=True,
                       help="Total number of jobs.")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input text files (.txt).")
    parser.add_argument("--num_samples", type=int, default=0,
                       help="Number of samples to process (0 = all files).")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Number of paragraphs to process in each batch.")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache embedding models.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.job_id < 0 or args.job_id >= args.total_jobs:
        raise ValueError(f"Job ID {args.job_id} must be between 0 and {args.total_jobs - 1}")
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Check if directory contains text files
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in input directory: {input_path}")
        sys.exit(1)
    
    print(f"Found {len(txt_files)} .txt files in input directory: {input_path}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA not available, will use CPU only")
    
    print(f"Using SPECTER 2 model: {SPECTER2_MODEL}")
    
    main(args)