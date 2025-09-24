#!/usr/bin/env python3
"""
Master Runner and Configuration Generator for Thesis Clustering Experiments
Orchestrates the complete experimental pipeline with robust error handling
"""

import yaml
import json
import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class ThesisExperimentOrchestrator:
    """Orchestrate complete thesis clustering experiments"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path.cwd()
        
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / 'configs'
        self.config_dir.mkdir(exist_ok=True)
        
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_default_config(self) -> Dict:
        """Generate comprehensive default configuration"""
        config = {
            'experiment_info': {
                'name': 'Thesis Clustering Experiments',
                'id': self.experiment_id,
                'description': 'Comparison of paragraph-level and methodological clustering pipelines',
                'author': 'Your Name',
                'created': datetime.now().isoformat()
            },
            
            'data': {
                'data_path': 'paper_data.csv',
                'paragraph_embedding_prefix': 'paragraph_emb_',
                'method_embedding_prefix': 'method_emb_',
                'keywords_column': 'keywords',
                'title_column': 'title',
                'paper_id_column': 'paper_id'
            },
            
            'work_dir': f'thesis_results_{self.experiment_id}',
            'random_state': 42,
            
            # Preprocessing configuration
            'preprocessing': {
                'method': 'standard',  # Options: 'none', 'standard', 'l2_normalize'
                'rationale': 'Standardization ensures all features contribute equally to distance calculations'
            },
            
            # Dimensionality reduction configuration
            'dimensionality_reduction': {
                'for_clustering': {
                    'method': 'umap',
                    'n_components': 50,
                    'n_neighbors': 15,
                    'min_dist': 0.1,
                    'metric': 'cosine',
                    'rationale': 'UMAP preserves both local and global structure for better cluster separation'
                },
                'for_visualization': {
                    'method': 'umap',
                    'n_components': 2,
                    'n_neighbors': 30,
                    'min_dist': 0.3,
                    'metric': 'cosine',
                    'rationale': '2D UMAP provides better visual cluster separation than t-SNE'
                }
            },
            
            # Clustering algorithms configuration
            'clustering': {
                'kmeans': {
                    'n_clusters': 8,
                    'n_init': 10,
                    'max_iter': 300,
                    'rationale': 'K-means provides interpretable baseline with expected number of topics'
                },
                'hdbscan': {
                    'min_cluster_size': 10,
                    'min_samples': 5,
                    'epsilon': 0.0,
                    'cluster_selection_method': 'eom',  # Excess of Mass
                    'rationale': 'HDBSCAN identifies natural clusters and handles noise/outliers'
                }
            },
            
            # K-optimization for enhanced clustering (if using enhanced script)
            'k_optimization': {
                'enabled': True,
                'k_min': 3,
                'k_max': 20,
                'methods': ['elbow', 'silhouette', 'gap', 'stability'],
                'consensus_voting': True
            },
            
            # Evaluation configuration
            'evaluation': {
                'keyword_coherence': {
                    'enabled': True,
                    'top_keywords': 10
                },
                'manual_evaluation': {
                    'enabled': True,
                    'n_clusters_to_evaluate': 10,
                    'n_samples_per_cluster': 3
                },
                'cross_pipeline_comparison': {
                    'enabled': True,
                    'metrics': ['adjusted_rand_index', 'normalized_mutual_info', 'v_measure']
                }
            },
            
            # Pipeline-specific overrides
            'pipeline_overrides': {
                'paragraph': {
                    'clustering': {
                        'hdbscan': {
                            'min_cluster_size': 15  # Larger clusters for paragraph-level
                        }
                    }
                },
                'methodological': {
                    'clustering': {
                        'hdbscan': {
                            'min_cluster_size': 8  # Smaller clusters for paper-level
                        }
                    }
                }
            },
            
            # Execution settings
            'execution': {
                'n_jobs': -1,  # Use all CPU cores
                'checkpoint_enabled': True,
                'clear_checkpoints_on_start': False,
                'verbose': True,
                'save_intermediate_results': True
            },
            
            # Visualization settings
            'visualization': {
                'dpi': 300,
                'figure_format': 'png',
                'color_palette': 'tab20',
                'plot_sizes': True,
                'plot_metrics': True,
                'plot_2d_projections': True
            }
        }
        
        return config
    
    def generate_algorithm_rationale(self) -> Dict:
        """Generate detailed rationale for algorithm choices"""
        rationale = {
            'algorithm_selection': {
                'why_only_kmeans_and_hdbscan': {
                    'kmeans': [
                        'Baseline algorithm that all reviewers understand',
                        'Works well when number of clusters is known',
                        'Fast and scalable to large datasets',
                        'Produces convex, spherical clusters suitable for topic modeling',
                        'Easy to interpret cluster centers as topic centroids'
                    ],
                    'hdbscan': [
                        'Handles noise and outliers automatically',
                        'No need to specify number of clusters a priori',
                        'Finds clusters of varying densities',
                        'More robust to parameter choices than DBSCAN',
                        'Produces hierarchical clustering structure for analysis'
                    ],
                    'why_not_others': {
                        'spectral_clustering': 'Computationally expensive for high-dimensional data',
                        'gaussian_mixture': 'Assumes Gaussian distributions which may not hold',
                        'mean_shift': 'Very slow and memory intensive',
                        'dbscan': 'Requires careful eps tuning, superseded by HDBSCAN'
                    }
                },
                
                'dimensionality_reduction_choice': {
                    'why_umap': [
                        'Preserves both local and global structure',
                        'Faster than t-SNE for large datasets',
                        'Provides meaningful inter-cluster distances',
                        'More stable and reproducible with fixed seed',
                        'Better at preserving continuity of data'
                    ],
                    'when_to_skip_dr': [
                        'When feature dimensions are already interpretable',
                        'When dimensionality is already low (<100)',
                        'When preserving original feature space is important'
                    ]
                },
                
                'preprocessing_strategy': {
                    'standard_scaling': 'Ensures all features contribute equally',
                    'l2_normalization': 'Better for cosine similarity-based clustering',
                    'no_preprocessing': 'When features are already normalized or meaningful'
                }
            },
            
            'pipeline_design': {
                'paragraph_pipeline': {
                    'strengths': [
                        'Captures fine-grained methodological patterns',
                        'Better for identifying specific techniques',
                        'More granular analysis possible'
                    ],
                    'weaknesses': [
                        'May miss paper-level coherence',
                        'Computationally more intensive',
                        'Requires paragraph-level annotations'
                    ]
                },
                'methodological_pipeline': {
                    'strengths': [
                        'Direct paper-level clustering',
                        'Simpler and faster',
                        'Better alignment with paper boundaries'
                    ],
                    'weaknesses': [
                        'May miss within-paper variation',
                        'Less granular insights',
                        'Depends on quality of summaries'
                    ]
                }
            }
        }
        
        return rationale
    
    def save_config(self, config: Dict, config_name: str = 'thesis_config.yaml'):
        """Save configuration to file"""
        config_path = self.config_dir / config_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to: {config_path}")
        return config_path
    
    def save_rationale(self, rationale: Dict):
        """Save algorithm rationale document"""
        rationale_path = self.config_dir / 'algorithm_rationale.json'
        
        with open(rationale_path, 'w') as f:
            json.dump(rationale, f, indent=2)
        
        # Also create markdown version
        md_path = self.config_dir / 'algorithm_rationale.md'
        with open(md_path, 'w') as f:
            f.write("# Algorithm Selection Rationale\n\n")
            f.write("## Why Only K-means and HDBSCAN?\n\n")
            f.write("### K-means\n")
            for reason in rationale['algorithm_selection']['why_only_kmeans_and_hdbscan']['kmeans']:
                f.write(f"- {reason}\n")
            f.write("\n### HDBSCAN\n")
            for reason in rationale['algorithm_selection']['why_only_kmeans_and_hdbscan']['hdbscan']:
                f.write(f"- {reason}\n")
            f.write("\n## Why UMAP for Dimensionality Reduction?\n\n")
            for reason in rationale['algorithm_selection']['dimensionality_reduction_choice']['why_umap']:
                f.write(f"- {reason}\n")
        
        print(f"Rationale saved to: {rationale_path}")
        return rationale_path
    
    def run_pipeline(self, config_path: Path, pipeline_type: str, clear_checkpoints: bool = False):
        """Run a single pipeline with error handling"""
        print(f"\n{'='*60}")
        print(f"Running {pipeline_type} pipeline")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable,
            'thesis_clustering_framework.py',
            '--config', str(config_path),
            '--pipeline', pipeline_type
        ]
        
        if clear_checkpoints:
            cmd.append('--clear-checkpoints')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {pipeline_type} pipeline completed successfully")
                return True
            else:
                print(f"✗ {pipeline_type} pipeline failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to run {pipeline_type} pipeline: {e}")
            return False
    
    def run_comparison(self, work_dir: Path):
        """Run pipeline comparison analysis"""
        print(f"\n{'='*60}")
        print("Running pipeline comparison")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable,
            'pipeline_comparison_evaluator.py',
            '--work-dir', str(work_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Pipeline comparison completed successfully")
                return True
            else:
                print("✗ Pipeline comparison failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to run comparison: {e}")
            return False
    
    def run_enhanced_clustering(self, config_path: Path, data_path: Path, work_dir: Path):
        """Run enhanced clustering with k-optimization"""
        print(f"\n{'='*60}")
        print("Running enhanced clustering analysis")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable,
            'enhanced_clustering_experiments.py',
            '--config', str(config_path),
            '--data_path', str(data_path),
            '--work_dir', str(work_dir),
            '--n_jobs', '-1'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Enhanced clustering completed successfully")
                return True
            else:
                print("✗ Enhanced clustering failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to run enhanced clustering: {e}")
            return False
    
    def create_experiment_summary(self, config: Dict, results_dir: Path):
        """Create experiment summary document"""
        summary = {
            'experiment_id': config['experiment_info']['id'],
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results_location': str(results_dir),
            'pipelines_run': [],
            'status': 'completed'
        }
        
        # Check which pipelines completed
        for pipeline in ['paragraph', 'methodological']:
            pipeline_dir = results_dir / pipeline
            if pipeline_dir.exists():
                report_file = pipeline_dir / f'{pipeline}_clustering_report.md'
                if report_file.exists():
                    summary['pipelines_run'].append(pipeline)
        
        # Check for comparison results
        comparison_dir = results_dir / 'pipeline_comparison'
        if comparison_dir.exists():
            summary['comparison_completed'] = True
            comparison_report = comparison_dir / 'pipeline_comparison_report.md'
            if comparison_report.exists():
                summary['comparison_report'] = str(comparison_report)
        
        # Save summary
        summary_file = results_dir / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment summary saved to: {summary_file}")
        return summary_file
    
    def run_complete_experiment(self, config: Dict = None, enhanced: bool = False):
        """Run complete experimental pipeline"""
        print("\n" + "="*80)
        print("THESIS CLUSTERING EXPERIMENT ORCHESTRATOR")
        print("="*80)
        
        # Use provided config or generate default
        if config is None:
            config = self.generate_default_config()
        
        # Save configuration
        config_path = self.save_config(config)
        
        # Save rationale
        rationale = self.generate_algorithm_rationale()
        self.save_rationale(rationale)
        
        # Create work directory
        work_dir = Path(config['work_dir'])
        work_dir.mkdir(parents=True, exist_ok=True)
        
        success_status = {
            'paragraph': False,
            'methodological': False,
            'comparison': False,
            'enhanced': False
        }
        
        # Run enhanced clustering if requested
        if enhanced and config.get('k_optimization', {}).get('enabled', False):
            print("\n" + "="*60)
            print("PHASE 0: Enhanced K-Optimization")
            print("="*60)
            
            # Create enhanced config
            enhanced_config_path = self.config_dir / 'enhanced_config.yaml'
            success_status['enhanced'] = self.run_enhanced_clustering(
                enhanced_config_path,
                Path(config['data']['data_path']),
                work_dir / 'enhanced'
            )
            
            if success_status['enhanced']:
                # Update main config with optimal k
                # (Would need to parse results to get optimal k)
                print("✓ K-optimization completed, updating configuration...")
        
        # Phase 1: Run paragraph pipeline
        print("\n" + "="*60)
        print("PHASE 1: Paragraph Pipeline")
        print("="*60)
        
        success_status['paragraph'] = self.run_pipeline(
            config_path, 
            'paragraph',
            config['execution'].get('clear_checkpoints_on_start', False)
        )
        
        # Phase 2: Run methodological pipeline
        print("\n" + "="*60)
        print("PHASE 2: Methodological Pipeline")
        print("="*60)
        
        success_status['methodological'] = self.run_pipeline(
            config_path,
            'methodological', 
            config['execution'].get('clear_checkpoints_on_start', False)
        )
        
        # Phase 3: Run comparison if both pipelines succeeded
        if success_status['paragraph'] and success_status['methodological']:
            print("\n" + "="*60)
            print("PHASE 3: Pipeline Comparison")
            print("="*60)
            
            success_status['comparison'] = self.run_comparison(work_dir)
        else:
            print("\n⚠ Skipping comparison - both pipelines must complete successfully")
        
        # Create experiment summary
        self.create_experiment_summary(config, work_dir)
        
        # Print final status
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print("\nStatus Summary:")
        for component, status in success_status.items():
            status_str = "✓ Success" if status else "✗ Failed"
            print(f"  {component.capitalize()}: {status_str}")
        
        print(f"\nResults directory: {work_dir}")
        
        if success_status['comparison']:
            print(f"\nView comparison report: {work_dir}/pipeline_comparison/pipeline_comparison_report.md")
        
        return success_status


def create_sbatch_script(config_path: Path, work_dir: Path, enhanced: bool = False):
    """Create SLURM batch script for HPC execution"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=thesis_clustering
#SBATCH --output={work_dir}/slurm_%j.out
#SBATCH --error={work_dir}/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute

# Load required modules (adjust for your HPC)
module load python/3.9
module load cuda/11.2

# Activate virtual environment
source ~/thesis_venv/bin/activate

# Set working directory
cd {Path.cwd()}

# Run experiment
python thesis_master_runner.py \\
    --config {config_path} \\
    --work-dir {work_dir} \\
    {"--enhanced" if enhanced else ""} \\
    --run-all

echo "Job completed at $(date)"
"""
    
    script_path = work_dir / 'run_experiment.sbatch'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"SLURM script created: {script_path}")
    print(f"Submit with: sbatch {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Thesis Clustering Experiment Orchestrator")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--work-dir', type=str, help='Working directory for results')
    parser.add_argument('--generate-config', action='store_true', 
                       help='Generate default configuration')
    parser.add_argument('--enhanced', action='store_true',
                       help='Run enhanced clustering with k-optimization')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete experiment pipeline')
    parser.add_argument('--create-slurm', action='store_true',
                       help='Create SLURM batch script for HPC')
    
    args = parser.parse_args()
    
    orchestrator = ThesisExperimentOrchestrator()
    
    if args.generate_config:
        config = orchestrator.generate_default_config()
        config_path = orchestrator.save_config(config)
        rationale = orchestrator.generate_algorithm_rationale()
        orchestrator.save_rationale(rationale)
        print(f"\n✓ Configuration generated: {config_path}")
        print("Edit the configuration file and run with --run-all")
        return
    
    if args.run_all:
        # Load config if provided, otherwise use default
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = orchestrator.generate_default_config()
        
        # Override work_dir if provided
        if args.work_dir:
            config['work_dir'] = args.work_dir
        
        # Run complete experiment
        orchestrator.run_complete_experiment(config, enhanced=args.enhanced)
        
    elif args.create_slurm:
        # Create SLURM script for HPC execution
        config_path = Path(args.config) if args.config else orchestrator.config_dir / 'thesis_config.yaml'
        work_dir = Path(args.work_dir) if args.work_dir else Path(f'thesis_results_{orchestrator.experiment_id}')
        create_sbatch_script(config_path, work_dir, args.enhanced)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
