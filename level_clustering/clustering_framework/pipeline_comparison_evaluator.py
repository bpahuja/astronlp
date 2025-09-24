#!/usr/bin/env python3
"""
Pipeline Comparison and Manual Evaluation Framework
For comparing paragraph vs methodological pipelines and conducting manual agreement studies
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    v_measure_score, fowlkes_mallows_score
)
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


class PipelineComparator:
    """Compare and evaluate different clustering pipelines"""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.comparison_dir = self.work_dir / 'pipeline_comparison'
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        self.paragraph_dir = self.work_dir / 'paragraph'
        self.method_dir = self.work_dir / 'methodological'
        
        self.results = {}
        self.evaluation_results = {}
    
    def load_pipeline_results(self, pipeline_type: str) -> Dict:
        """Load results from a pipeline"""
        pipeline_dir = self.work_dir / pipeline_type / 'checkpoints'
        
        results = {}
        
        # Load clustering results
        clustering_file = pipeline_dir / 'noise_reassignment.pkl'
        if clustering_file.exists():
            with open(clustering_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results['clustering'] = checkpoint_data['data']
                results['clustering_metadata'] = checkpoint_data.get('metadata', {})
        
        # Load evaluation results
        eval_file = pipeline_dir / 'evaluation.pkl'
        if eval_file.exists():
            with open(eval_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results['evaluation'] = checkpoint_data['data']
        
        # Load representatives
        rep_file = pipeline_dir / 'representatives.pkl'
        if rep_file.exists():
            with open(rep_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results['representatives'] = checkpoint_data['data']
        
        return results
    
    def compare_clustering_quality(self, paragraph_results: Dict, method_results: Dict) -> Dict:
        """Compare clustering quality metrics between pipelines"""
        comparison = {
            'paragraph_pipeline': {},
            'methodological_pipeline': {},
            'comparison_metrics': {}
        }
        
        # Extract metrics for each pipeline
        for pipeline_name, results in [('paragraph_pipeline', paragraph_results), 
                                       ('methodological_pipeline', method_results)]:
            if 'clustering' not in results:
                continue
            
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                if algorithm in results['clustering']:
                    alg_data = results['clustering'][algorithm]
                    
                    # Get labels
                    labels = alg_data['labels']
                    unique_labels = np.unique(labels[labels >= 0])
                    
                    comparison[pipeline_name][algorithm] = {
                        'n_clusters': len(unique_labels),
                        'cluster_sizes': pd.Series(labels[labels >= 0]).value_counts().to_dict(),
                        'size_variance': np.var(pd.Series(labels[labels >= 0]).value_counts()),
                        'min_cluster_size': pd.Series(labels[labels >= 0]).value_counts().min(),
                        'max_cluster_size': pd.Series(labels[labels >= 0]).value_counts().max()
                    }
                    
                    # Add metrics if available
                    if algorithm == 'kmeans' and 'kmeans' in results['clustering']:
                        comparison[pipeline_name][algorithm].update(
                            results['clustering']['kmeans'].get('metrics', {})
                        )
        
        # Statistical comparison
        if paragraph_results.get('clustering') and method_results.get('clustering'):
            # Compare cluster size distributions
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                if algorithm in paragraph_results['clustering'] and algorithm in method_results['clustering']:
                    para_labels = paragraph_results['clustering'][algorithm]['labels']
                    meth_labels = method_results['clustering'][algorithm]['labels']
                    
                    # Ensure same length for comparison
                    min_len = min(len(para_labels), len(meth_labels))
                    if min_len > 0:
                        # Cross-pipeline agreement
                        ari = adjusted_rand_score(para_labels[:min_len], meth_labels[:min_len])
                        nmi = normalized_mutual_info_score(para_labels[:min_len], meth_labels[:min_len])
                        v_measure = v_measure_score(para_labels[:min_len], meth_labels[:min_len])
                        
                        comparison['comparison_metrics'][f'{algorithm}_agreement'] = {
                            'adjusted_rand_index': ari,
                            'normalized_mutual_info': nmi,
                            'v_measure': v_measure
                        }
        
        return comparison
    
    def evaluate_keyword_coherence_comparison(self, paragraph_results: Dict, 
                                             method_results: Dict) -> Dict:
        """Compare keyword coherence between pipelines"""
        coherence_comparison = {}
        
        for pipeline_name, results in [('paragraph', paragraph_results),
                                       ('methodological', method_results)]:
            if 'evaluation' in results:
                eval_data = results['evaluation']
                
                coherence_comparison[pipeline_name] = {}
                for algorithm in ['kmeans', 'hdbscan_reassigned']:
                    if algorithm in eval_data:
                        coherence_comparison[pipeline_name][algorithm] = {
                            'mean_coherence': eval_data[algorithm].get('mean_coherence', 0),
                            'std_coherence': eval_data[algorithm].get('std_coherence', 0)
                        }
        
        # Statistical test for difference
        if 'paragraph' in coherence_comparison and 'methodological' in coherence_comparison:
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                if algorithm in coherence_comparison['paragraph'] and algorithm in coherence_comparison['methodological']:
                    para_coherence = coherence_comparison['paragraph'][algorithm]['mean_coherence']
                    meth_coherence = coherence_comparison['methodological'][algorithm]['mean_coherence']
                    
                    coherence_comparison[f'{algorithm}_difference'] = {
                        'absolute_diff': abs(para_coherence - meth_coherence),
                        'percent_diff': ((para_coherence - meth_coherence) / max(para_coherence, 0.001)) * 100,
                        'better_pipeline': 'paragraph' if para_coherence > meth_coherence else 'methodological'
                    }
        
        return coherence_comparison
    
    def create_comparison_visualizations(self, comparison_results: Dict, coherence_results: Dict):
        """Create comprehensive comparison visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Number of clusters comparison
        ax1 = plt.subplot(2, 4, 1)
        pipelines = ['paragraph_pipeline', 'methodological_pipeline']
        algorithms = ['kmeans', 'hdbscan_reassigned']
        
        data_clusters = []
        labels_clusters = []
        for pipeline in pipelines:
            if pipeline in comparison_results:
                for algorithm in algorithms:
                    if algorithm in comparison_results[pipeline]:
                        data_clusters.append(comparison_results[pipeline][algorithm].get('n_clusters', 0))
                        labels_clusters.append(f"{pipeline.split('_')[0][:4]}\n{algorithm[:6]}")
        
        if data_clusters:
            colors = ['skyblue', 'lightblue', 'lightcoral', 'salmon']
            bars = ax1.bar(range(len(data_clusters)), data_clusters, color=colors[:len(data_clusters)])
            ax1.set_xticks(range(len(labels_clusters)))
            ax1.set_xticklabels(labels_clusters, fontsize=8)
            ax1.set_ylabel('Number of Clusters')
            ax1.set_title('Cluster Count Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, data_clusters):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(value), ha='center', va='bottom')
        
        # 2. Silhouette score comparison
        ax2 = plt.subplot(2, 4, 2)
        silhouette_data = []
        silhouette_labels = []
        
        for pipeline in pipelines:
            if pipeline in comparison_results:
                for algorithm in algorithms:
                    if algorithm in comparison_results[pipeline]:
                        score = comparison_results[pipeline][algorithm].get('silhouette', 0)
                        if score and score > -1:
                            silhouette_data.append(score)
                            silhouette_labels.append(f"{pipeline.split('_')[0][:4]}\n{algorithm[:6]}")
        
        if silhouette_data:
            bars = ax2.bar(range(len(silhouette_data)), silhouette_data, 
                          color=['gold', 'yellow', 'orange', 'coral'][:len(silhouette_data)])
            ax2.set_xticks(range(len(silhouette_labels)))
            ax2.set_xticklabels(silhouette_labels, fontsize=8)
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Score Comparison')
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, silhouette_data):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Cross-pipeline agreement heatmap
        ax3 = plt.subplot(2, 4, 3)
        if 'comparison_metrics' in comparison_results:
            agreement_data = []
            for algorithm in algorithms:
                key = f'{algorithm}_agreement'
                if key in comparison_results['comparison_metrics']:
                    metrics = comparison_results['comparison_metrics'][key]
                    agreement_data.append([
                        metrics.get('adjusted_rand_index', 0),
                        metrics.get('normalized_mutual_info', 0),
                        metrics.get('v_measure', 0)
                    ])
            
            if agreement_data:
                agreement_array = np.array(agreement_data)
                im = ax3.imshow(agreement_array, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
                ax3.set_xticks(range(3))
                ax3.set_xticklabels(['ARI', 'NMI', 'V-measure'], fontsize=9)
                ax3.set_yticks(range(len(algorithms)))
                ax3.set_yticklabels([a[:6] for a in algorithms])
                ax3.set_title('Cross-Pipeline Agreement')
                plt.colorbar(im, ax=ax3)
                
                # Add text annotations
                for i in range(len(algorithms)):
                    for j in range(3):
                        if i < len(agreement_data) and j < len(agreement_data[i]):
                            ax3.text(j, i, f'{agreement_data[i][j]:.2f}',
                                   ha='center', va='center', color='white' if agreement_data[i][j] < 0.5 else 'black')
        
        # 4. Keyword coherence comparison
        ax4 = plt.subplot(2, 4, 4)
        if coherence_results:
            coherence_data = []
            coherence_labels = []
            
            for pipeline in ['paragraph', 'methodological']:
                if pipeline in coherence_results:
                    for algorithm in algorithms:
                        if algorithm in coherence_results[pipeline]:
                            coherence = coherence_results[pipeline][algorithm].get('mean_coherence', 0)
                            coherence_data.append(coherence)
                            coherence_labels.append(f"{pipeline[:4]}\n{algorithm[:6]}")
            
            if coherence_data:
                bars = ax4.bar(range(len(coherence_data)), coherence_data,
                             color=['lightgreen', 'green', 'darkgreen', 'olive'][:len(coherence_data)])
                ax4.set_xticks(range(len(coherence_labels)))
                ax4.set_xticklabels(coherence_labels, fontsize=8)
                ax4.set_ylabel('Mean Coherence')
                ax4.set_title('Keyword Coherence Comparison')
                ax4.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, coherence_data):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Cluster size distribution - Paragraph pipeline
        ax5 = plt.subplot(2, 4, 5)
        if 'paragraph_pipeline' in comparison_results:
            for algorithm in ['kmeans']:
                if algorithm in comparison_results['paragraph_pipeline']:
                    sizes = comparison_results['paragraph_pipeline'][algorithm].get('cluster_sizes', {})
                    if sizes:
                        sorted_sizes = sorted(sizes.values(), reverse=True)
                        ax5.bar(range(len(sorted_sizes)), sorted_sizes, color='skyblue', alpha=0.7)
                        ax5.set_xlabel('Cluster Rank')
                        ax5.set_ylabel('Size')
                        ax5.set_title(f'Paragraph Pipeline - {algorithm} Sizes')
                        ax5.grid(True, alpha=0.3)
        
        # 6. Cluster size distribution - Methodological pipeline
        ax6 = plt.subplot(2, 4, 6)
        if 'methodological_pipeline' in comparison_results:
            for algorithm in ['kmeans']:
                if algorithm in comparison_results['methodological_pipeline']:
                    sizes = comparison_results['methodological_pipeline'][algorithm].get('cluster_sizes', {})
                    if sizes:
                        sorted_sizes = sorted(sizes.values(), reverse=True)
                        ax6.bar(range(len(sorted_sizes)), sorted_sizes, color='lightcoral', alpha=0.7)
                        ax6.set_xlabel('Cluster Rank')
                        ax6.set_ylabel('Size')
                        ax6.set_title(f'Methodological Pipeline - {algorithm} Sizes')
                        ax6.grid(True, alpha=0.3)
        
        # 7. Summary metrics table
        ax7 = plt.subplot(2, 4, 7)
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create summary table
        summary_data = []
        summary_data.append(['Metric', 'Paragraph', 'Methodological', 'Better'])
        
        # Add key comparison metrics
        metrics_to_compare = ['silhouette', 'n_clusters', 'mean_coherence']
        for metric in metrics_to_compare:
            para_val = None
            meth_val = None
            
            if metric == 'mean_coherence':
                if 'paragraph' in coherence_results and 'kmeans' in coherence_results['paragraph']:
                    para_val = coherence_results['paragraph']['kmeans'].get('mean_coherence', 0)
                if 'methodological' in coherence_results and 'kmeans' in coherence_results['methodological']:
                    meth_val = coherence_results['methodological']['kmeans'].get('mean_coherence', 0)
            else:
                if 'paragraph_pipeline' in comparison_results and 'kmeans' in comparison_results['paragraph_pipeline']:
                    para_val = comparison_results['paragraph_pipeline']['kmeans'].get(metric)
                if 'methodological_pipeline' in comparison_results and 'kmeans' in comparison_results['methodological_pipeline']:
                    meth_val = comparison_results['methodological_pipeline']['kmeans'].get(metric)
            
            if para_val is not None and meth_val is not None:
                if isinstance(para_val, float):
                    para_str = f'{para_val:.3f}'
                    meth_str = f'{meth_val:.3f}'
                else:
                    para_str = str(para_val)
                    meth_str = str(meth_val)
                
                if metric in ['silhouette', 'mean_coherence']:
                    better = 'Para' if para_val > meth_val else 'Meth'
                else:
                    better = '-'
                
                summary_data.append([metric, para_str, meth_str, better])
        
        table = ax7.table(cellText=summary_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('Summary Comparison', fontsize=12, fontweight='bold', pad=20)
        
        # 8. Agreement scores comparison
        ax8 = plt.subplot(2, 4, 8)
        if 'comparison_metrics' in comparison_results:
            agreement_scores = []
            agreement_labels = []
            
            for algorithm in algorithms:
                key = f'{algorithm}_agreement'
                if key in comparison_results['comparison_metrics']:
                    metrics = comparison_results['comparison_metrics'][key]
                    # Average of all agreement metrics
                    avg_agreement = np.mean([
                        metrics.get('adjusted_rand_index', 0),
                        metrics.get('normalized_mutual_info', 0),
                        metrics.get('v_measure', 0)
                    ])
                    agreement_scores.append(avg_agreement)
                    agreement_labels.append(algorithm.replace('_reassigned', '\n(reass.)'))
            
            if agreement_scores:
                bars = ax8.bar(range(len(agreement_scores)), agreement_scores,
                             color=['purple', 'indigo'][:len(agreement_scores)], alpha=0.7)
                ax8.set_xticks(range(len(agreement_labels)))
                ax8.set_xticklabels(agreement_labels)
                ax8.set_ylabel('Average Agreement Score')
                ax8.set_title('Pipeline Agreement by Algorithm')
                ax8.set_ylim(0, 1)
                ax8.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, agreement_scores):
                    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Pipeline Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.comparison_dir / 'pipeline_comparison_visualization.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualizations saved to {fig_path}")
        return fig_path
    
    def create_manual_evaluation_template(self, paragraph_results: Dict, method_results: Dict,
                                         n_samples_per_cluster: int = 3):
        """Create template for manual evaluation of cluster quality"""
        evaluation_template = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'n_samples_per_cluster': n_samples_per_cluster,
                'evaluator': None,
                'evaluation_criteria': {
                    'coherence': 'How well do papers in the cluster share a common theme?',
                    'separation': 'How distinct is this cluster from others?',
                    'quality': 'Overall cluster quality (1-5 scale)'
                }
            },
            'paragraph_pipeline': {},
            'methodological_pipeline': {}
        }
        
        # Process each pipeline
        for pipeline_name, results in [('paragraph_pipeline', paragraph_results),
                                       ('methodological_pipeline', method_results)]:
            if 'representatives' not in results:
                continue
            
            pipeline_eval = {}
            
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                if algorithm in results['representatives']:
                    algorithm_eval = {}
                    
                    representatives = results['representatives'][algorithm]
                    
                    # Sample clusters for evaluation
                    cluster_ids = list(representatives.keys())
                    sampled_clusters = np.random.choice(
                        cluster_ids, 
                        min(10, len(cluster_ids)),  # Evaluate up to 10 clusters
                        replace=False
                    )
                    
                    for cluster_id in sampled_clusters:
                        cluster_info = representatives[cluster_id]
                        
                        algorithm_eval[cluster_id] = {
                            'representative': cluster_info,
                            'evaluation': {
                                'coherence_score': None,  # To be filled manually (1-5)
                                'separation_score': None,  # To be filled manually (1-5)
                                'quality_score': None,     # To be filled manually (1-5)
                                'keywords': None,          # To be filled manually
                                'theme_description': None, # To be filled manually
                                'notes': None              # Optional notes
                            }
                        }
                    
                    pipeline_eval[algorithm] = algorithm_eval
            
            evaluation_template[pipeline_name] = pipeline_eval
        
        # Save template
        template_file = self.comparison_dir / 'manual_evaluation_template.json'
        with open(template_file, 'w') as f:
            json.dump(evaluation_template, f, indent=2)
        
        print(f"Manual evaluation template saved to {template_file}")
        return evaluation_template
    
    def analyze_manual_evaluations(self, evaluation_file: Path) -> Dict:
        """Analyze completed manual evaluations"""
        with open(evaluation_file, 'r') as f:
            evaluations = json.load(f)
        
        analysis = {
            'paragraph_pipeline': {},
            'methodological_pipeline': {},
            'comparison': {}
        }
        
        # Analyze each pipeline
        for pipeline_name in ['paragraph_pipeline', 'methodological_pipeline']:
            if pipeline_name not in evaluations:
                continue
            
            pipeline_scores = {
                'kmeans': {'coherence': [], 'separation': [], 'quality': []},
                'hdbscan_reassigned': {'coherence': [], 'separation': [], 'quality': []}
            }
            
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                if algorithm in evaluations[pipeline_name]:
                    for cluster_id, cluster_eval in evaluations[pipeline_name][algorithm].items():
                        eval_scores = cluster_eval.get('evaluation', {})
                        
                        for metric in ['coherence', 'separation', 'quality']:
                            score_key = f'{metric}_score'
                            if score_key in eval_scores and eval_scores[score_key] is not None:
                                pipeline_scores[algorithm][metric].append(eval_scores[score_key])
            
            # Compute statistics
            pipeline_stats = {}
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                algorithm_stats = {}
                for metric in ['coherence', 'separation', 'quality']:
                    scores = pipeline_scores[algorithm][metric]
                    if scores:
                        algorithm_stats[metric] = {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'median': np.median(scores),
                            'min': np.min(scores),
                            'max': np.max(scores),
                            'n_evaluated': len(scores)
                        }
                pipeline_stats[algorithm] = algorithm_stats
            
            analysis[pipeline_name] = pipeline_stats
        
        # Statistical comparison between pipelines
        comparison = {}
        
        for algorithm in ['kmeans', 'hdbscan_reassigned']:
            algorithm_comparison = {}
            
            for metric in ['coherence', 'separation', 'quality']:
                para_scores = []
                meth_scores = []
                
                if ('paragraph_pipeline' in analysis and 
                    algorithm in analysis['paragraph_pipeline'] and
                    metric in analysis['paragraph_pipeline'][algorithm]):
                    para_scores = pipeline_scores[algorithm][metric] if 'paragraph_pipeline' in locals() else []
                
                if ('methodological_pipeline' in analysis and 
                    algorithm in analysis['methodological_pipeline'] and
                    metric in analysis['methodological_pipeline'][algorithm]):
                    meth_scores = pipeline_scores[algorithm][metric] if 'methodological_pipeline' in locals() else []
                
                if para_scores and meth_scores:
                    # Mann-Whitney U test
                    statistic, p_value = mannwhitneyu(para_scores, meth_scores)
                    
                    algorithm_comparison[metric] = {
                        'mann_whitney_u': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'better_pipeline': 'paragraph' if np.mean(para_scores) > np.mean(meth_scores) else 'methodological',
                        'effect_size': abs(np.mean(para_scores) - np.mean(meth_scores))
                    }
            
            if algorithm_comparison:
                comparison[algorithm] = algorithm_comparison
        
        analysis['comparison'] = comparison
        
        # Save analysis
        analysis_file = self.comparison_dir / 'manual_evaluation_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def generate_final_report(self, comparison_results: Dict, coherence_results: Dict,
                            manual_analysis: Optional[Dict] = None):
        """Generate comprehensive comparison report"""
        report_lines = [
            "# Thesis Clustering Pipeline Comparison Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Determine overall winner
        para_wins = 0
        meth_wins = 0
        
        # Check silhouette scores
        if 'paragraph_pipeline' in comparison_results and 'methodological_pipeline' in comparison_results:
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                para_sil = comparison_results['paragraph_pipeline'].get(algorithm, {}).get('silhouette', -1)
                meth_sil = comparison_results['methodological_pipeline'].get(algorithm, {}).get('silhouette', -1)
                if para_sil > meth_sil:
                    para_wins += 1
                elif meth_sil > para_sil:
                    meth_wins += 1
        
        # Check coherence
        for algorithm in ['kmeans', 'hdbscan_reassigned']:
            if f'{algorithm}_difference' in coherence_results:
                if coherence_results[f'{algorithm}_difference']['better_pipeline'] == 'paragraph':
                    para_wins += 1
                else:
                    meth_wins += 1
        
        winner = 'Paragraph' if para_wins > meth_wins else 'Methodological' if meth_wins > para_wins else 'Tie'
        
        report_lines.extend([
            f"**Overall Better Pipeline: {winner}**",
            f"- Paragraph wins: {para_wins} metrics",
            f"- Methodological wins: {meth_wins} metrics",
            "",
            "## 1. Clustering Quality Comparison",
            ""
        ])
        
        # Add clustering metrics comparison
        if 'paragraph_pipeline' in comparison_results and 'methodological_pipeline' in comparison_results:
            report_lines.append("| Algorithm | Metric | Paragraph | Methodological | Better |")
            report_lines.append("|-----------|--------|-----------|----------------|--------|")
            
            for algorithm in ['kmeans', 'hdbscan_reassigned']:
                for metric in ['n_clusters', 'silhouette']:
                    para_val = comparison_results['paragraph_pipeline'].get(algorithm, {}).get(metric)
                    meth_val = comparison_results['methodological_pipeline'].get(algorithm, {}).get(metric)
                    
                    if para_val is not None and meth_val is not None:
                        if isinstance(para_val, float):
                            para_str = f"{para_val:.3f}"
                            meth_str = f"{meth_val:.3f}"
                            better = "Para" if para_val > meth_val else "Meth"
                        else:
                            para_str = str(para_val)
                            meth_str = str(meth_val)
                            better = "-"
                        
                        report_lines.append(f"| {algorithm} | {metric} | {para_str} | {meth_str} | {better} |")
        
        report_lines.extend([
            "",
            "## 2. Cross-Pipeline Agreement",
            ""
        ])
        
        if 'comparison_metrics' in comparison_results:
            for algorithm in ['kmeans_agreement', 'hdbscan_reassigned_agreement']:
                if algorithm in comparison_results['comparison_metrics']:
                    metrics = comparison_results['comparison_metrics'][algorithm]
                    report_lines.append(f"### {algorithm.replace('_', ' ').title()}")
                    report_lines.append(f"- Adjusted Rand Index: {metrics.get('adjusted_rand_index', 0):.3f}")
                    report_lines.append(f"- Normalized Mutual Info: {metrics.get('normalized_mutual_info', 0):.3f}")
                    report_lines.append(f"- V-Measure: {metrics.get('v_measure', 0):.3f}")
                    report_lines.append("")
        
        report_lines.extend([
            "## 3. Keyword Coherence Analysis",
            ""
        ])
        
        for pipeline in ['paragraph', 'methodological']:
            if pipeline in coherence_results:
                report_lines.append(f"### {pipeline.title()} Pipeline")
                for algorithm in ['kmeans', 'hdbscan_reassigned']:
                    if algorithm in coherence_results[pipeline]:
                        coh = coherence_results[pipeline][algorithm]
                        report_lines.append(f"- {algorithm}: {coh.get('mean_coherence', 0):.3f} ± {coh.get('std_coherence', 0):.3f}")
                report_lines.append("")
        
        # Add manual evaluation results if available
        if manual_analysis:
            report_lines.extend([
                "## 4. Manual Evaluation Results",
                ""
            ])
            
            for pipeline_name in ['paragraph_pipeline', 'methodological_pipeline']:
                if pipeline_name in manual_analysis:
                    report_lines.append(f"### {pipeline_name.replace('_', ' ').title()}")
                    
                    for algorithm in ['kmeans', 'hdbscan_reassigned']:
                        if algorithm in manual_analysis[pipeline_name]:
                            alg_stats = manual_analysis[pipeline_name][algorithm]
                            
                            report_lines.append(f"**{algorithm}:**")
                            for metric in ['coherence', 'separation', 'quality']:
                                if metric in alg_stats:
                                    stats = alg_stats[metric]
                                    report_lines.append(f"- {metric.title()}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['n_evaluated']})")
                            report_lines.append("")
            
            if 'comparison' in manual_analysis:
                report_lines.append("### Statistical Comparison")
                for algorithm in ['kmeans', 'hdbscan_reassigned']:
                    if algorithm in manual_analysis['comparison']:
                        report_lines.append(f"**{algorithm}:**")
                        for metric in ['coherence', 'separation', 'quality']:
                            if metric in manual_analysis['comparison'][algorithm]:
                                comp = manual_analysis['comparison'][algorithm][metric]
                                report_lines.append(f"- {metric.title()}: p={comp['p_value']:.4f}, "
                                                  f"better={comp['better_pipeline']}")
                        report_lines.append("")
        
        report_lines.extend([
            "## 5. Recommendations",
            "",
            "### Algorithm Choice",
            "- **K-means**: Suitable when you need a fixed number of interpretable clusters",
            "- **HDBSCAN**: Better for discovering natural cluster structure with noise handling",
            "",
            "### Pipeline Selection",
        ])
        
        if winner == 'Paragraph':
            report_lines.append("- **Recommended: Paragraph Pipeline**")
            report_lines.append("  - Better captures fine-grained methodological patterns")
            report_lines.append("  - Higher clustering quality metrics")
        elif winner == 'Methodological':
            report_lines.append("- **Recommended: Methodological Pipeline**")
            report_lines.append("  - Better captures paper-level themes")
            report_lines.append("  - More coherent cluster assignments")
        else:
            report_lines.append("- **Both pipelines show similar performance**")
            report_lines.append("  - Consider ensemble approach combining both")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.comparison_dir / 'pipeline_comparison_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Final comparison report saved to {report_file}")
        return report_file
    
    def run_complete_comparison(self):
        """Run complete comparison analysis"""
        print("\n" + "="*80)
        print("RUNNING PIPELINE COMPARISON ANALYSIS")
        print("="*80)
        
        # Load results from both pipelines
        print("\nLoading pipeline results...")
        paragraph_results = self.load_pipeline_results('paragraph')
        method_results = self.load_pipeline_results('methodological')
        
        if not paragraph_results or not method_results:
            print("Warning: Could not load results from both pipelines")
            print("Make sure both pipelines have been run first")
            return
        
        # Compare clustering quality
        print("\nComparing clustering quality...")
        comparison_results = self.compare_clustering_quality(paragraph_results, method_results)
        
        # Compare keyword coherence
        print("\nComparing keyword coherence...")
        coherence_results = self.evaluate_keyword_coherence_comparison(paragraph_results, method_results)
        
        # Create visualizations
        print("\nCreating comparison visualizations...")
        self.create_comparison_visualizations(comparison_results, coherence_results)
        
        # Create manual evaluation template
        print("\nCreating manual evaluation template...")
        self.create_manual_evaluation_template(paragraph_results, method_results)
        
        # Generate final report
        print("\nGenerating final comparison report...")
        report_path = self.generate_final_report(comparison_results, coherence_results)
        
        # Save all comparison results
        results_file = self.comparison_dir / 'comparison_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'comparison': comparison_results,
                'coherence': coherence_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("COMPARISON ANALYSIS COMPLETED")
        print("="*80)
        print(f"Results saved in: {self.comparison_dir}")
        
        return {
            'comparison': comparison_results,
            'coherence': coherence_results,
            'report_path': report_path
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Comparison Framework")
    parser.add_argument('--work-dir', type=str, required=True, 
                       help='Working directory containing pipeline results')
    parser.add_argument('--evaluate', type=str, 
                       help='Path to completed manual evaluation JSON file')
    
    args = parser.parse_args()
    
    comparator = PipelineComparator(Path(args.work_dir))
    
    if args.evaluate:
        # Analyze manual evaluations
        print("Analyzing manual evaluations...")
        manual_analysis = comparator.analyze_manual_evaluations(Path(args.evaluate))
        
        # Regenerate report with manual evaluation results
        comparison_file = comparator.comparison_dir / 'comparison_results.json'
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                saved_results = json.load(f)
            
            comparator.generate_final_report(
                saved_results.get('comparison', {}),
                saved_results.get('coherence', {}),
                manual_analysis
            )
    else:
        # Run full comparison
        comparator.run_complete_comparison()
    
    print("\n✓ Comparison analysis completed!")


if __name__ == "__main__":
    main()
