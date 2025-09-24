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
import threading
import queue
import signal
import atexit
from typing import List, Tuple, Dict, Any, Optional
import random
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

# Set multiprocessing start method for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Force output flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default models
DEFAULT_MODELS = [
    "BAAI/bge-large-en-v1.5",
    "kwang2049/TSDAE-twitterpara",
    "intfloat/e5-large-v2"
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "sentence-transformers/all-mpnet-base-v2",
]

class FaultTolerantEmbedder:
    """Fault-tolerant multi-GPU embedder optimized for large-scale processing."""
    
    def __init__(self, args):
        self.args = args
        self.models_to_process = self.get_models_to_process()
        
        # Base directories
        self.work_dir = Path("data/embeddings_work")
        self.embeddings_base_dir = Path("data/embeddings")
        self.checkpoints_dir = self.work_dir / "checkpoints"
        self.progress_dir = self.work_dir / "progress"
        
        # Create directories
        for dir_path in [self.work_dir, self.embeddings_base_dir, self.checkpoints_dir, self.progress_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device_queue = queue.Queue()
        for i in range(self.num_gpus):
            self.device_queue.put(i)
        
        # Memory management
        self.memory_threshold = 0.85
        self.max_workers = min(self.num_gpus * 2, os.cpu_count()) if self.num_gpus > 0 else os.cpu_count()
        
        # Progress tracking
        self.global_progress_db = self.progress_dir / "global_progress.db"
        self.setup_global_progress_db()
        
        # Fault tolerance
        self.checkpoint_interval = 100  # Save checkpoint every N files
        self.last_checkpoint_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        atexit.register(self.cleanup_on_exit)
        
        self.shutdown_event = threading.Event()
        
        self.flush_print("Multi-GPU Fault-Tolerant Embedder initialized")
        self.flush_print(f"Available GPUs: {self.num_gpus}")
        self.flush_print(f"Max workers: {self.max_workers}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.flush_print(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def cleanup_on_exit(self):
        """Cleanup on exit."""
        self.flush_print("Cleaning up on exit...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_models_to_process(self):
        """Get list of models to process."""
        if self.args.models:
            return self.args.models
        else:
            return DEFAULT_MODELS
    
    def get_model_name_safe(self, model_name):
        """Get safe model name for directory creation."""
        return model_name.replace("/", "*")
    
    def setup_global_progress_db(self):
        """Setup SQLite database for global progress tracking."""
        conn = sqlite3.connect(self.global_progress_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS file_progress (
                file_path TEXT,
                model_name TEXT,
                status TEXT,
                timestamp REAL,
                error_msg TEXT,
                worker_id TEXT,
                gpu_id INTEGER,
                PRIMARY KEY (file_path, model_name)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS job_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_job_progress(self, model_name: str) -> Dict[str, str]:
        """Get progress for a specific model."""
        conn = sqlite3.connect(self.global_progress_db)
        cursor = conn.execute(
            "SELECT file_path, status FROM file_progress WHERE model_name = ?",
            (model_name,)
        )
        progress = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return progress
    
    def mark_file_progress(self, file_path: str, model_name: str, status: str, 
                          error_msg: str = None, worker_id: str = None, gpu_id: int = None):
        """Mark file processing progress."""
        conn = sqlite3.connect(self.global_progress_db)
        conn.execute('''
            INSERT OR REPLACE INTO file_progress 
            (file_path, model_name, status, timestamp, error_msg, worker_id, gpu_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_path, model_name, status, time.time(), error_msg, worker_id, gpu_id))
        conn.commit()
        conn.close()
    
    def save_checkpoint(self, model_name: str, completed_batches: List[Dict]):
        """Save checkpoint for fault tolerance."""
        checkpoint_file = self.checkpoints_dir / f"{self.get_model_name_safe(model_name)}_checkpoint.json"
        checkpoint_data = {
            'model_name': model_name,
            'completed_batches': completed_batches,
            'timestamp': time.time(),
            'total_batches': len(completed_batches)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        self.flush_print(f"Checkpoint saved for {model_name}: {len(completed_batches)} batches completed")
    
    def load_checkpoint(self, model_name: str) -> Dict[str, Any]:
        """Load checkpoint for resuming."""
        checkpoint_file = self.checkpoints_dir / f"{self.get_model_name_safe(model_name)}_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def flush_print(self, message):
        """Print with immediate flush."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
        sys.stdout.flush()
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0
    
    def cleanup_memory(self):
        """Force garbage collection and CUDA cache cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def init_model_on_device(self, model_name: str, device_id: int):
        """Initialize embedding model on specific GPU device."""
        try:
            # Set cache directory
            if self.args.cache_dir:
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.args.cache_dir)
            
            # Initialize model
            if device_id >= 0 and torch.cuda.is_available():
                device = f'cuda:{device_id}'
                torch.cuda.set_device(device_id)
            else:
                device = 'cpu'
                device_id = -1
            
            model = SentenceTransformer(model_name, cache_folder=self.args.cache_dir, device=device)
            
            self.flush_print(f"Model {model_name} loaded on device {device}")
            return model, device_id
                
        except Exception as e:
            self.flush_print(f"Error initializing model {model_name} on device {device_id}: {e}")
            return None, -1
    
    def generate_embeddings_batch(self, texts: List[str], paper_ids: List[str], model, model_name: str, device_id: int):
        """Generate embeddings for a batch of texts."""
        try:
            # Set device for processing
            if device_id >= 0:
                torch.cuda.set_device(device_id)
            
            # Generate embeddings
            embeddings = model.encode(
                texts,
                batch_size=min(self.args.batch_size, len(texts)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=f'cuda:{device_id}' if device_id >= 0 else 'cpu'
            )
            
            # Package results
            results = []
            for i, (paper_id, embedding) in enumerate(zip(paper_ids, embeddings)):
                results.append({
                    'paper_id': paper_id,
                    'embedding': embedding,
                    'model_name': model_name,
                    'device_id': device_id,
                    'embedding_dim': embedding.shape[0]
                })
            
            return results
            
        except Exception as e:
            self.flush_print(f"Error generating embeddings: {e}")
            return []
    
    def save_embeddings_batch(self, embeddings_dir: Path, batch_results: List[Dict], batch_idx: int):
        """Save batch of embeddings."""
        batch_file = embeddings_dir / f"batch_{batch_idx:06d}.npz"
        
        # Extract embeddings and metadata
        embeddings = np.array([result['embedding'] for result in batch_results])
        paper_ids = [result['paper_id'] for result in batch_results]
        
        # Save as compressed numpy file
        np.savez_compressed(
            batch_file,
            embeddings=embeddings,
            paper_ids=paper_ids,
            model_name=batch_results[0]['model_name'] if batch_results else "",
            embedding_dim=batch_results[0]['embedding_dim'] if batch_results else 0
        )
        
        return len(batch_results)
    
    def process_file_batch(self, file_batch: List[Tuple[Path, str]], model_name: str, 
                          batch_idx: int, worker_id: str) -> Dict[str, Any]:
        """Process a batch of files on a single GPU."""
        if self.shutdown_event.is_set():
            return {'success': False, 'batch_idx': batch_idx, 'count': 0}
        
        # Get GPU device
        try:
            device_id = self.device_queue.get(timeout=60)
        except queue.Empty:
            self.flush_print(f"Worker {worker_id}: No GPU available, skipping batch {batch_idx}")
            return {'success': False, 'batch_idx': batch_idx, 'count': 0}
        
        try:
            # Initialize model on device
            model, actual_device_id = self.init_model_on_device(model_name, device_id)
            
            if model is None:
                self.flush_print(f"Worker {worker_id}: Failed to initialize model {model_name}")
                return {'success': False, 'batch_idx': batch_idx, 'count': 0}
            
            # Setup output directory
            model_name_safe = self.get_model_name_safe(model_name)
            embeddings_dir = self.embeddings_base_dir / model_name_safe
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Load texts
            texts = []
            paper_ids = []
            valid_files = []
            
            for txt_file, paper_id in file_batch:
                if self.shutdown_event.is_set():
                    break
                
                try:
                    # Read document
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        document_text = f.read().strip()
                    
                    if document_text:
                        texts.append(document_text)
                        paper_ids.append(paper_id)
                        valid_files.append(txt_file)
                    else:
                        self.mark_file_progress(str(txt_file), model_name, 'empty', 
                                              worker_id=worker_id, gpu_id=actual_device_id)
                        
                except Exception as e:
                    error_msg = f"Error reading {paper_id}: {str(e)}"
                    self.mark_file_progress(str(txt_file), model_name, 'error', 
                                          error_msg=error_msg, worker_id=worker_id, 
                                          gpu_id=actual_device_id)
            
            if not texts:
                return {'success': False, 'batch_idx': batch_idx, 'count': 0}
            
            # Generate embeddings
            batch_results = self.generate_embeddings_batch(
                texts, paper_ids, model, model_name, actual_device_id
            )
            
            if batch_results:
                # Save embeddings batch
                saved_count = self.save_embeddings_batch(embeddings_dir, batch_results, batch_idx)
                
                # Mark files as completed
                for txt_file in valid_files:
                    self.mark_file_progress(str(txt_file), model_name, 'completed',
                                          worker_id=worker_id, gpu_id=actual_device_id)
                
                return {
                    'success': True, 
                    'batch_idx': batch_idx, 
                    'count': saved_count,
                    'embeddings': batch_results
                }
            else:
                return {'success': False, 'batch_idx': batch_idx, 'count': 0}
        
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            self.flush_print(f"Worker {worker_id}: {error_msg}")
            
            # Mark batch files as failed
            for txt_file, paper_id in file_batch:
                self.mark_file_progress(str(txt_file), model_name, 'error', 
                                      error_msg=error_msg, worker_id=worker_id, 
                                      gpu_id=device_id)
            
            return {'success': False, 'batch_idx': batch_idx, 'count': 0}
        
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            self.cleanup_memory()
            
            # Return GPU to queue
            self.device_queue.put(device_id)
    
    def consolidate_embeddings(self, model_name: str, all_results: List[Dict]):
        """Consolidate all batch results into final embedding files."""
        model_name_safe = self.get_model_name_safe(model_name)
        embeddings_dir = self.embeddings_base_dir / model_name_safe
        
        # Collect all embeddings and paper IDs
        all_embeddings = []
        all_paper_ids = []
        
        for result in all_results:
            if result['success'] and 'embeddings' in result:
                for embed_data in result['embeddings']:
                    all_embeddings.append(embed_data['embedding'])
                    all_paper_ids.append(embed_data['paper_id'])
        
        if not all_embeddings:
            self.flush_print(f"No embeddings to consolidate for {model_name}")
            return
        
        # Convert to numpy array
        embeddings_matrix = np.array(all_embeddings)
        
        # Save consolidated embeddings
        embed_save_path = embeddings_dir / "embeddings.npy"
        map_save_path = embeddings_dir / "paperid_to_idx.json"
        
        np.save(embed_save_path, embeddings_matrix)
        
        # Create paper ID to index mapping
        paperid_to_idx = {pid: idx for idx, pid in enumerate(all_paper_ids)}
        with open(map_save_path, 'w') as f:
            json.dump(paperid_to_idx, f, indent=2)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'num_papers': len(all_paper_ids),
            'embedding_dim': embeddings_matrix.shape[1],
            'embedding_shape': list(embeddings_matrix.shape),
            'timestamp': time.time()
        }
        
        metadata_path = embeddings_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.flush_print(f"Consolidated embeddings for {model_name}:")
        self.flush_print(f"  Shape: {embeddings_matrix.shape}")
        self.flush_print(f"  Saved to: {embed_save_path}")
        self.flush_print(f"  Mapping: {map_save_path}")
    
    def process_model_optimized(self, txt_files: List[Path], model_name: str):
        """Process all files for a single model using all available GPUs."""
        self.flush_print(f"Starting optimized embedding processing for {model_name} with {len(txt_files)} files")
        
        # Load checkpoint and progress
        checkpoint = self.load_checkpoint(model_name)
        progress = self.get_job_progress(model_name)
        
        # Filter out already completed files
        remaining_files = []
        for txt_file in txt_files:
            file_path = str(txt_file)
            if progress.get(file_path) != 'completed':
                remaining_files.append(txt_file)
        
        completed_count = len(txt_files) - len(remaining_files)
        self.flush_print(f"Model {model_name}: {completed_count} already completed, {len(remaining_files)} remaining")
        
        if not remaining_files:
            self.flush_print(f"All files already processed for {model_name}")
            return
        
        # Shuffle files for better load balancing
        random.shuffle(remaining_files)
        
        # Create batches
        batch_size = max(1, min(self.args.batch_size, len(remaining_files) // (self.num_gpus * 4)))
        batches = []
        for i in range(0, len(remaining_files), batch_size):
            batch_files = remaining_files[i:i + batch_size]
            batch = [(f, f.stem) for f in batch_files]
            batches.append(batch)
        
        self.flush_print(f"Model {model_name}: Created {len(batches)} batches of size {batch_size}")
        
        # Process batches in parallel
        completed_batches = []
        all_results = []
        last_checkpoint = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_file_batch, batch, model_name, i, f"worker_{i}"): (batch, i)
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                if self.shutdown_event.is_set():
                    break
                
                try:
                    batch_result = future.result()
                    batch, batch_idx = future_to_batch[future]
                    
                    if batch_result['success']:
                        completed_batches.append(batch_result)
                        all_results.append(batch_result)
                    
                    # Update progress
                    total_completed = completed_count + sum(r['count'] for r in completed_batches)
                    progress_pct = len(completed_batches) / len(batches) * 100
                    self.flush_print(f"Model {model_name}: Batch {batch_idx+1}/{len(batches)} ({progress_pct:.1f}%) - {total_completed} files processed")
                    
                    # Save checkpoint periodically
                    if (time.time() - last_checkpoint > 300 or  # Every 5 minutes
                        len(completed_batches) % self.checkpoint_interval == 0):
                        self.save_checkpoint(model_name, [{'batch_idx': r['batch_idx'], 'count': r['count']} for r in completed_batches])
                        last_checkpoint = time.time()
                    
                    # Memory check
                    if self.get_memory_usage() > self.memory_threshold:
                        self.flush_print(f"Memory usage high ({self.get_memory_usage():.1%}), cleaning up...")
                        self.cleanup_memory()
                        
                except Exception as e:
                    batch, batch_idx = future_to_batch[future]
                    self.flush_print(f"Batch {batch_idx} processing error: {e}")
        
        # Final checkpoint
        self.save_checkpoint(model_name, [{'batch_idx': r['batch_idx'], 'count': r['count']} for r in completed_batches])
        
        # Consolidate all embeddings
        self.consolidate_embeddings(model_name, all_results)
        
        final_completed = sum(r['count'] for r in completed_batches)
        self.flush_print(f"Model {model_name} completed: {final_completed} embeddings generated")
    
    def load_summary_files(self) -> List[Path]:
        """Load all summary files from the specified directory."""
        summary_root = Path(self.args.summary_root)
        if not summary_root.exists():
            raise FileNotFoundError(f"Summary directory not found: {summary_root}")
        
        txt_files = list(summary_root.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {summary_root}")
        
        # Limit samples if requested
        if self.args.num_samples > 0:
            txt_files = txt_files[:self.args.num_samples]
        
        self.flush_print(f"Loaded {len(txt_files)} summary files for embedding")
        return txt_files
    
    def estimate_completion_time(self, txt_files: List[Path]) -> None:
        """Estimate completion time based on system resources."""
        total_files = len(txt_files)
        num_models = len(self.models_to_process)
        
        # Rough estimates (files per minute per GPU)
        files_per_minute_per_gpu = 200  # Embeddings are generally faster than summarization
        
        total_minutes = 0
        for model_name in self.models_to_process:
            model_minutes = total_files / (files_per_minute_per_gpu * max(1, self.num_gpus))
            total_minutes += model_minutes
        
        hours = total_minutes / 60
        self.flush_print(f"Estimated completion time: {hours:.1f} hours ({total_minutes:.0f} minutes)")
    
    def run(self):
        """Main execution method."""
        self.flush_print("=" * 80)
        self.flush_print("STARTING FAULT-TOLERANT MULTI-GPU EMBEDDING GENERATION")
        self.flush_print("=" * 80)
        
        # Load summary files
        txt_files = self.load_summary_files()
        
        # Print configuration
        self.flush_print(f"Summary root: {self.args.summary_root}")
        self.flush_print(f"Total files to process: {len(txt_files)}")
        self.flush_print(f"Models to process: {len(self.models_to_process)}")
        for i, model in enumerate(self.models_to_process):
            self.flush_print(f"  {i+1}. {model}")
        self.flush_print(f"Available GPUs: {self.num_gpus}")
        self.flush_print(f"Max workers: {self.max_workers}")
        self.flush_print(f"Batch size: {self.args.batch_size}")
        
        # Estimate completion time
        self.estimate_completion_time(txt_files)
        
        # Process each model
        for model_idx, model_name in enumerate(self.models_to_process):
            if self.shutdown_event.is_set():
                self.flush_print("Shutdown signal received, stopping processing")
                break
            
            self.flush_print(f"Processing model {model_idx + 1}/{len(self.models_to_process)}: {model_name}")
            
            try:
                self.process_model_optimized(txt_files, model_name)
            except Exception as e:
                self.flush_print(f"Error processing model {model_name}: {e}")
                continue
            
            # Cleanup between models
            self.cleanup_memory()
            time.sleep(5)  # Brief pause
        
        self.flush_print("=" * 80)
        self.flush_print("EMBEDDING GENERATION COMPLETE")
        self.flush_print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fault-Tolerant Multi-GPU Embedding Generation")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                       help="List of embedding model identifiers to process")
    parser.add_argument("--summary_root", type=str, required=True,
                       help="Root directory containing summary text files")
    parser.add_argument("--num_samples", type=int, default=0,
                       help="Number of samples to process (0 = all files)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for embedding generation")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache models")
    
    args = parser.parse_args()
    
    # Validate GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA not available, will use CPU only")
    
    # Validate summary root directory
    if not Path(args.summary_root).exists():
        print(f"ERROR: Summary root directory does not exist: {args.summary_root}")
        sys.exit(1)
    
    # Initialize and run embedder
    embedder = FaultTolerantEmbedder(args)
    embedder.run()


if __name__ == "__main__":
    main()