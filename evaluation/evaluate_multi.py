import os
import argparse
import json
import pandas as pd
import sys
import wandb
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import time
import sqlite3
import shutil
import math
import gc
import psutil

# Evaluation metric imports
from evaluate import load as load_metric
from summac.model_summac import SummaCConv
from summac.benchmark import SummaCBenchmark

# Force output flushing for SLURM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# --- UTILITY FOR EXTRACTIVE FRAGMENTS ---
def calculate_extractive_fragments(summary, source):
    """
    Calculates the extractive fragment coverage and density of a summary against a source text.
    Coverage: Percentage of summary words that are in extractive fragments.
    Density: Average length of extractive fragments.
    """
    summary_tokens = summary.lower().split()
    source_lower = source.lower()

    if not summary_tokens:
        return 0.0, 0.0

    extractive_fragments = []
    current_fragment = []

    for token in summary_tokens:
        if token in source_lower:  # Simplified check
            current_fragment.append(token)
        else:
            if current_fragment:
                extractive_fragments.append(" ".join(current_fragment))
                current_fragment = []

    if current_fragment:
        extractive_fragments.append(" ".join(current_fragment))

    fragment_lengths = [len(frag.split()) for frag in extractive_fragments]
    total_fragment_words = sum(fragment_lengths)

    coverage = total_fragment_words / len(summary_tokens) if summary_tokens else 0
    density = sum(fragment_lengths) / len(fragment_lengths) if fragment_lengths else 0

    return coverage, density


def load_text_file(file_path):
    """Load text from a file, returning empty string if file doesn't exist or can't be read."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ""


def evaluate_single_summary(summary_data, use_gpu=False):
    """
    Evaluate a single summary. This function will be called in parallel.
    """
    try:
        # Initialize metrics for this process
        rouge_scorer = load_metric('rouge')
        bert_scorer = load_metric('bertscore')
        
        # Initialize SummaC for this process
        summac_conv = SummaCConv(
            models=["vitc"], 
            bins='percentile', 
            granularity="sentence", 
            nli_labels="e",
            device="cuda" if use_gpu else "cpu", 
            start_file="default", 
            agg="mean"
        )
        
        paper_id = summary_data['paper_id']
        source_text = summary_data['source_text']
        generated_summary = summary_data['generated_summary']
        reference_summary = summary_data['reference_summary']
        
        result = {
            'paper_id': paper_id,
            'generated_summary': generated_summary,
            'source_text': source_text,
            'reference_summary': reference_summary
        }
        
        # Evaluate against source text if available
        if source_text.strip():
            # ROUGE (vs. Source)
            rouge_scores = rouge_scorer.compute(
                predictions=[generated_summary], 
                references=[source_text]
            )
            result.update({
                'rouge1_vs_source': rouge_scores['rouge1'],
                'rouge2_vs_source': rouge_scores['rouge2'],
                'rougeL_vs_source': rouge_scores['rougeL']
            })
            
            # BERTScore (vs. Source)
            bert_scores = bert_scorer.compute(
                predictions=[generated_summary], 
                references=[source_text], 
                lang="en"
            )
            result['bertscore_f1_vs_source'] = bert_scores['f1'][0]
            
            # SummaC (Consistency with source)
            summac_score = summac_conv.score([source_text], [generated_summary])['scores'][0]
            result['summac_score'] = summac_score
            
            # Extractive Fragments
            coverage, density = calculate_extractive_fragments(generated_summary, source_text)
            result.update({
                'extractive_coverage': coverage,
                'extractive_density': density
            })
        
        # Evaluate against reference summary if available
        if reference_summary.strip():
            rouge_scores_ref = rouge_scorer.compute(
                predictions=[generated_summary], 
                references=[reference_summary]
            )
            result.update({
                'rouge1_vs_reference': rouge_scores_ref['rouge1'],
                'rouge2_vs_reference': rouge_scores_ref['rouge2'],
                'rougeL_vs_reference': rouge_scores_ref['rougeL']
            })
            
            bert_scores_ref = bert_scorer.compute(
                predictions=[generated_summary], 
                references=[reference_summary], 
                lang="en"
            )
            result['bertscore_f1_vs_reference'] = bert_scores_ref['f1'][0]
        
        return result
        
    except Exception as e:
        print(f"Error evaluating summary for paper {summary_data.get('paper_id', 'unknown')}: {e}", flush=True)
        return {
            'paper_id': summary_data.get('paper_id', 'unknown'),
            'evaluation_error': str(e)
        }


class MultiJobEvaluator:
    """Multi-job evaluator with controller-worker pattern for evaluation tasks."""
    
    def __init__(self, args):
        self.args = args
        self.job_id = args.job_id
        
        # Base directories
        self.work_dir = Path("data/eval_work_v1")
        self.splits_dir = self.work_dir / "splits"
        self.flags_dir = self.work_dir / "flags"
        
        # Create base directories
        for dir_path in [self.work_dir, self.splits_dir, self.flags_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Memory management
        self.memory_threshold = 0.8
        self.batch_size = args.batch_size
        
        # Global flag files for coordination
        self.data_split_done_flag = self.flags_dir / "data_split_done"
        self.controller_done_flag = self.flags_dir / "controller_done"
        self.worker_done_flag = self.flags_dir / f"worker_{self.job_id}_done"
        self.all_done_flag = self.flags_dir / "all_jobs_done"
        
        # Model-specific paths
        model_dir = args.model_name.replace("/", "*")
        self.summaries_dir = Path(args.summaries_dir) / model_dir
        self.methodology_dir = Path(args.methodology_dir)
        self.results_dir = self.summaries_dir / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0
    
    def cleanup_memory(self):
        """Force garbage collection and CUDA cache cleanup."""
        gc.collect()
        import torch
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
    
    def setup_progress_db(self):
        """Setup SQLite database for tracking progress."""
        job_results_dir = self.results_dir / f"job_{self.job_id}"
        job_results_dir.mkdir(parents=True, exist_ok=True)
        
        progress_db = job_results_dir / "progress.db"
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluated_papers (
                paper_id TEXT PRIMARY KEY,
                status TEXT,
                timestamp REAL,
                error_msg TEXT
            )
        ''')
        conn.commit()
        conn.close()
        return progress_db
    
    def get_evaluated_papers(self, progress_db):
        """Get set of already evaluated paper IDs."""
        conn = sqlite3.connect(progress_db)
        cursor = conn.execute("SELECT paper_id FROM evaluated_papers WHERE status = 'completed'")
        evaluated = {row[0] for row in cursor.fetchall()}
        conn.close()
        return evaluated
    
    def mark_paper_evaluated(self, progress_db, paper_id, status='completed', error_msg=None):
        """Mark a paper as evaluated in the database."""
        conn = sqlite3.connect(progress_db)
        conn.execute('''
            INSERT OR REPLACE INTO evaluated_papers (paper_id, status, timestamp, error_msg)
            VALUES (?, ?, ?, ?)
        ''', (paper_id, status, time.time(), error_msg))
        conn.commit()
        conn.close()
    
    def load_summaries_and_sources(self):
        """
        Load summaries and their corresponding source texts.
        Returns a list of dictionaries with paper_id, generated_summary, and source_text.
        """
        if not self.summaries_dir.exists():
            raise FileNotFoundError(f"Summaries directory not found: {self.summaries_dir}")
        
        if not self.methodology_dir.exists():
            raise FileNotFoundError(f"Methodology dataset directory not found: {self.methodology_dir}")
        
        # Get all summary files
        summary_files = list(self.summaries_dir.glob("*.txt"))
        self.flush_print(f"Found {len(summary_files)} summary files")
        
        summaries = []
        missing_source_count = 0
        
        for summary_file in summary_files:
            paper_id = summary_file.stem
            
            # Load generated summary
            generated_summary = load_text_file(summary_file)
            if not generated_summary:
                continue
            
            # Load corresponding source text
            source_file = self.methodology_dir / f"{paper_id}.txt"
            source_text = load_text_file(source_file)
            
            if not source_text:
                missing_source_count += 1
                self.flush_print(f"Warning: No source text found for {paper_id}")
            
            summaries.append({
                'paper_id': paper_id,
                'generated_summary': generated_summary,
                'source_text': source_text,
                'reference_summary': ''  # No reference summaries in this setup
            })
        
        self.flush_print(f"Loaded {len(summaries)} summaries")
        self.flush_print(f"Missing source texts: {missing_source_count}")
        
        return summaries
    
    def split_data(self):
        """Split data into chunks for processing (Controller Job 0 only)."""
        self.flush_print("Starting data splitting...")
        
        # Load all summaries
        summaries = self.load_summaries_and_sources()
        
        if not summaries:
            raise FileNotFoundError(f"No summaries found in {self.summaries_dir}")
        
        self.flush_print(f"Found {len(summaries)} total summaries")
        
        # Limit samples if requested
        if self.args.num_samples > 0:
            summaries = summaries[:self.args.num_samples]
            self.flush_print(f"Limited to first {self.args.num_samples} summaries")
        
        # Calculate splits
        total_jobs = self.args.total_jobs
        summaries_per_job = math.ceil(len(summaries) / total_jobs)
        
        self.flush_print(f"Splitting {len(summaries)} summaries into {total_jobs} jobs ({summaries_per_job} summaries per job)")
        
        # Create splits
        for job_id in range(total_jobs):
            start_idx = job_id * summaries_per_job
            end_idx = min((job_id + 1) * summaries_per_job, len(summaries))
            job_summaries = summaries[start_idx:end_idx]
            
            if job_summaries:  # Only create split if there are summaries
                split_file = self.splits_dir / f"job_{job_id}_summaries.json"
                with open(split_file, 'w') as f:
                    json.dump(job_summaries, f, indent=2)
                self.flush_print(f"Created split for job {job_id}: {len(job_summaries)} summaries")
        
        # Create data split done flag
        self.create_flag(self.data_split_done_flag)
        self.flush_print("Data splitting complete")
    
    def load_job_summaries(self):
        """Load summaries assigned to this job."""
        split_file = self.splits_dir / f"job_{self.job_id}_summaries.json"
        
        if not split_file.exists():
            self.flush_print(f"No split file found for job {self.job_id}")
            return []
        
        with open(split_file, 'r') as f:
            summaries = json.load(f)
        
        self.flush_print(f"Loaded {len(summaries)} summaries for job {self.job_id}")
        return summaries
    
    def evaluate_batch(self, summary_batch):
        """Evaluate a batch of summaries."""
        # Create partial function with use_gpu parameter
        evaluate_func = partial(evaluate_single_summary, use_gpu=self.args.use_gpu)
        
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.args.num_workers) as executor:
            # Submit all tasks
            future_to_summary = {
                executor.submit(evaluate_func, summary_data): summary_data 
                for summary_data in summary_batch
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_summary), total=len(summary_batch), desc="Evaluating batch"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    summary_data = future_to_summary[future]
                    self.flush_print(f"Error processing {summary_data.get('paper_id', 'unknown')}: {e}")
                    results.append({
                        'paper_id': summary_data.get('paper_id', 'unknown'),
                        'evaluation_error': str(e)
                    })
        
        return results
    
    def run_evaluation(self, summaries):
        """Run evaluation for assigned summaries."""
        if not summaries:
            self.flush_print("No summaries to evaluate")
            return
        
        self.flush_print(f"Starting evaluation with {len(summaries)} summaries")
        
        # Setup progress tracking
        progress_db = self.setup_progress_db()
        
        # Check how many are already evaluated
        evaluated = self.get_evaluated_papers(progress_db)
        remaining_summaries = [s for s in summaries if s['paper_id'] not in evaluated]
        
        self.flush_print(f"Already evaluated: {len(evaluated)}")
        self.flush_print(f"Remaining to evaluate: {len(remaining_summaries)}")
        
        if not remaining_summaries:
            self.flush_print("All summaries already evaluated!")
            return
        
        # Create batches
        batches = []
        for i in range(0, len(remaining_summaries), self.batch_size):
            batch = remaining_summaries[i:i + self.batch_size]
            batches.append(batch)
        
        self.flush_print(f"Created {len(batches)} batches")
        
        # Evaluate batches
        all_results = []
        completed_count = len(evaluated)
        total_summaries = len(summaries)
        
        for i, batch in enumerate(batches):
            self.flush_print(f"Processing batch {i+1}/{len(batches)}")
            
            try:
                results = self.evaluate_batch(batch)
                all_results.extend(results)
                
                # Mark papers as evaluated
                for result in results:
                    paper_id = result.get('paper_id')
                    if 'evaluation_error' in result:
                        self.mark_paper_evaluated(progress_db, paper_id, 'error', result['evaluation_error'])
                    else:
                        self.mark_paper_evaluated(progress_db, paper_id, 'completed')
                
                completed_count += len(results)
                
            except Exception as e:
                self.flush_print(f"Error processing batch {i+1}: {e}")
                continue
            
            # Memory check
            if self.get_memory_usage() > self.memory_threshold:
                self.flush_print(f"Memory usage high ({self.get_memory_usage():.1%}), cleaning up...")
                self.cleanup_memory()
            
            # Progress update
            self.flush_print(f"Progress: {completed_count}/{total_summaries} ({completed_count/total_summaries:.1%})")
        
        # Save job results
        job_results_dir = self.results_dir / f"job_{self.job_id}"
        job_results_file = job_results_dir / "results.csv"
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(job_results_file, index=False)
            self.flush_print(f"Saved {len(all_results)} results to {job_results_file}")
        
        self.flush_print(f"Evaluation complete. Processed {completed_count} summaries.")
    
    def consolidate_results(self):
        """Consolidate results from all jobs (Controller Job 0 only)."""
        self.flush_print("Consolidating results from all jobs...")
        
        # Wait for all worker jobs to complete
        for job_id in range(1, self.args.total_jobs):
            worker_flag = self.flags_dir / f"worker_{job_id}_done"
            if worker_flag.exists():
                self.flush_print(f"Worker {job_id} completed")
        
        # Collect all results
        all_results = []
        for job_id in range(self.args.total_jobs):
            job_results_file = self.results_dir / f"job_{job_id}" / "results.csv"
            if job_results_file.exists():
                job_df = pd.read_csv(job_results_file)
                all_results.append(job_df)
                self.flush_print(f"Loaded {len(job_df)} results from job {job_id}")
        
        if not all_results:
            self.flush_print("No results found to consolidate")
            return
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        self.flush_print(f"Combined {len(combined_df)} total results")
        
        # Filter out results with errors for statistics calculation
        valid_results = combined_df[~combined_df['evaluation_error'].notna()] if 'evaluation_error' in combined_df.columns else combined_df
        
        # Calculate summary statistics
        summary_metrics = {}
        
        # Metrics vs source
        if 'rouge1_vs_source' in valid_results.columns and len(valid_results) > 0:
            summary_metrics.update({
                'avg_rouge1_vs_source': valid_results['rouge1_vs_source'].mean(),
                'avg_rouge2_vs_source': valid_results['rouge2_vs_source'].mean(),
                'avg_rougeL_vs_source': valid_results['rougeL_vs_source'].mean(),
                'avg_bertscore_f1_vs_source': valid_results['bertscore_f1_vs_source'].mean(),
                'avg_summac': valid_results['summac_score'].mean(),
                'avg_extractive_coverage': valid_results['extractive_coverage'].mean(),
                'avg_extractive_density': valid_results['extractive_density'].mean(),
            })
        
        # Metrics vs reference
        if 'rouge1_vs_reference' in valid_results.columns and len(valid_results) > 0:
            summary_metrics.update({
                'avg_rouge1_vs_reference': valid_results['rouge1_vs_reference'].mean(),
                'avg_rouge2_vs_reference': valid_results['rouge2_vs_reference'].mean(),
                'avg_rougeL_vs_reference': valid_results['rougeL_vs_reference'].mean(),
                'avg_bertscore_f1_vs_reference': valid_results['bertscore_f1_vs_reference'].mean(),
            })
        
        # Add metadata
        summary_metrics.update({
            'total_summaries': len(combined_df),
            'valid_summaries': len(valid_results),
            'failed_evaluations': len(combined_df) - len(valid_results)
        })
        
        # Save consolidated results
        final_results_file = self.summaries_dir / "evaluation_results.csv"
        combined_df.to_csv(final_results_file, index=False)
        self.flush_print(f"Saved consolidated results to: {final_results_file}")
        
        # Save summary metrics
        metrics_file = self.summaries_dir / "summary_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        self.flush_print(f"Saved summary metrics to: {metrics_file}")
        
        # Log to wandb (offline mode)
        self.flush_print("Logging results to wandb (offline mode)...")
        
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(
            project=self.args.project_name,
            job_type="evaluation",
            name=f"eval_{self.args.model_name}",
            config={
                'summaries_dir': self.args.summaries_dir,
                'methodology_dir': self.args.methodology_dir,
                'model_name': self.args.model_name,
                'num_workers': self.args.num_workers,
                'use_gpu': self.args.use_gpu,
                'num_summaries_evaluated': len(combined_df),
                'num_valid_summaries': len(valid_results),
                'total_jobs': self.args.total_jobs
            }
        )
        
        # Log summary metrics
        wandb.log(summary_metrics)
        
        # Log the full results table
        wandb.log({"evaluation_results": wandb.Table(dataframe=combined_df)})
        
        run.finish()
        
        # Print summary statistics
        self.flush_print("\n--- SUMMARY STATISTICS ---")
        for metric, value in summary_metrics.items():
            if isinstance(value, float):
                self.flush_print(f"{metric}: {value:.4f}")
            else:
                self.flush_print(f"{metric}: {value}")
        
        self.flush_print(f"Results logged to local wandb directory: {run.dir}")
        self.flush_print("To sync with the cloud, run:")
        self.flush_print(f"wandb sync {run.dir}")
        
        self.create_flag(self.all_done_flag)
        self.flush_print("Results consolidation complete!")
    
    def run(self):
        """Main execution method."""
        self.flush_print("=" * 60)
        self.flush_print("STARTING MULTI-JOB EVALUATION")
        self.flush_print("=" * 60)
        self.flush_print(f"Model: {self.args.model_name}")
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
            
            # Step 2: Evaluate own summaries
            summaries = self.load_job_summaries()
            self.run_evaluation(summaries)
            
            # Step 3: Mark controller done
            self.create_flag(self.controller_done_flag)
            
            # Step 4: Wait for all workers and consolidate
            self.consolidate_results()
            
        else:
            # Worker job (Job 1+)
            self.flush_print("Running as WORKER job")
            
            # Step 1: Wait for data split
            self.wait_for_flag(self.data_split_done_flag)
            
            # Step 2: Evaluate assigned summaries
            summaries = self.load_job_summaries()
            self.run_evaluation(summaries)
            
            # Step 3: Mark worker done
            self.create_flag(self.worker_done_flag)
        
        self.flush_print("=" * 60)
        self.flush_print("JOB COMPLETE")
        self.flush_print("=" * 60)


def main(args):
    """Main function using the multi-job evaluator."""
    evaluator = MultiJobEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Job Summary Evaluation with Controller-Worker Pattern")
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="data/summaries_v6",
        help="Directory containing generated summaries (default: data/summaries_v6)"
    )
    parser.add_argument(
        "--methodology_dir",
        type=str,
        default="data/methodology_dataset",
        help="Directory containing original methodology texts (default: data/methodology_dataset)"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="summarization-evaluation",
        help="Name of the wandb project"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count() // 4,
        help=f"Number of parallel workers per job (default: {mp.cpu_count() // 4})"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for evaluation metrics (faster but requires CUDA)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model used for summaries (e.g., facebook/bart-large)"
    )
    parser.add_argument(
        "--job_id", 
        type=int, 
        required=True,
        help="Job ID (0 = controller, 1+ = worker)."
    )
    parser.add_argument(
        "--total_jobs", 
        type=int, 
        required=True,
        help="Total number of jobs."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=0,
        help="Number of samples to evaluate (0 = all summaries)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=50,
        help="Number of summaries to evaluate in each batch."
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.job_id < 0 or args.job_id >= args.total_jobs:
        raise ValueError(f"Job ID {args.job_id} must be between 0 and {args.total_jobs - 1}")
    
    main(args)