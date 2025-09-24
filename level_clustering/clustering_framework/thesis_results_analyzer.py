#!/usr/bin/env python3
"""
Results Analysis and Reporting Utilities for Thesis Clustering
Provides comprehensive analysis tools and LaTeX/publication-ready outputs
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-paper')


class ThesisResultsAnalyzer:
    """Comprehensive results analysis and reporting"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir / 'final_analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.latex_dir = self.analysis_dir / 'latex'
        self.latex_dir.mkdir(exist_ok=True)
        
        self.figures_dir = self.analysis_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
    
    def generate_latex_tables(self, paragraph_results: Dict, method_results: Dict,
                            comparison_results: Dict) -> List[str]:
        """Generate publication-ready LaTeX tables"""
        latex_tables = []
        
        # Table 1: Pipeline Performance Comparison
        table1 = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Clustering Performance Comparison Between Pipelines}",
            "\\label{tab:pipeline_comparison}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Pipeline & Algorithm & Clusters & Silhouette & Calinski-H & Davies-B \\\\",
            "\\midrule"
        ]
        
        # Add data rows
        for pipeline_name, results in [('Paragraph', paragraph_results), 
                                       ('Methodological', method_results)]:
            if 'clustering' in results:
                for alg in ['kmeans', 'hdbscan_reassigned']:
                    if alg in results['clustering'] and 'metrics' in results['clustering'][alg]:
                        metrics = results['clustering'][alg]['metrics']
                        table1.append(
                            f"{pipeline_name} & {alg.replace('_reassigned', '')} & "
                            f"{metrics.get('n_clusters', '-')} & "
                            f"{metrics.get('silhouette', -1):.3f} & "
                            f"{metrics.get('calinski_harabasz', 0):.1f} & "
                            f"{metrics.get('davies_bouldin', float('inf')):.3f} \\\\"
                        )
        
        table1.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        latex_tables.append('\n'.join(table1))
        
        # Table 2: Cross-Pipeline Agreement Metrics
        if comparison_results and 'comparison_metrics' in comparison_results:
            table2 = [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Cross-Pipeline Agreement Metrics}",
                "\\label{tab:agreement_metrics}",
                "\\begin{tabular}{lccc}",
                "\\toprule",
                "Algorithm & ARI & NMI & V-Measure \\\\",
                "\\midrule"
            ]
            
            for alg_key in ['kmeans_agreement', 'hdbscan_reassigned_agreement']:
                if alg_key in comparison_results['comparison_metrics']:
                    metrics = comparison_results['comparison_metrics'][alg_key]
                    alg_name = alg_key.replace('_agreement', '').replace('_reassigned', '')
                    table2.append(
                        f"{alg_name} & "
                        f"{metrics.get('adjusted_rand_index', 0):.3f} & "
                        f"{metrics.get('normalized_mutual_info', 0):.3f} & "
                        f"{metrics.get('v_measure', 0):.3f} \\\\"
                    )
            
            table2.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
            
            latex_tables.append('\n'.join(table2))
        
        # Table 3: Noise Reassignment Statistics
        table3 = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{HDBSCAN Noise Reassignment Statistics}",
            "\\label{tab:noise_reassignment}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Pipeline & Original Noise (\\%) & Reassigned Points \\\\",
            "\\midrule"
        ]
        
        for pipeline_name, results in [('Paragraph', paragraph_results),
                                       ('Methodological', method_results)]:
            if 'clustering' in results and 'hdbscan' in results['clustering']:
                noise_ratio = results['clustering']['hdbscan']['metrics'].get('noise_ratio', 0)
                if 'hdbscan_reassigned' in results['clustering']:
                    n_reassigned = results['clustering'].get('hdbscan_reassigned', {}).get('n_reassigned', 0)
                else:
                    n_reassigned = 0
                
                table3.append(
                    f"{pipeline_name} & {noise_ratio*100:.1f}\\% & {n_reassigned} \\\\"
                )
        
        table3.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        latex_tables.append('\n'.join(table3))
        
        # Save all tables
        latex_file = self.latex_dir / 'tables.tex'
        with open(latex_file, 'w') as f:
            f.write("% Automatically generated LaTeX tables\n")
            f.write("% Include in your thesis with \\input{tables.tex}\n\n")
            f.write('\n\n'.join(latex_tables))
        
        print(f"LaTeX tables saved to: {latex_file}")
        return latex_tables
    
    def create_publication_figures(self, paragraph_results: Dict, method_results: Dict):
        """Create publication-quality figures"""
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12
        })
        
        # Figure 1: Comparative metrics bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Silhouette scores comparison
        algorithms = ['K-means', 'HDBSCAN']
        pipelines = ['Paragraph', 'Methodological']
        
        silhouette_data = []
        for pipeline_name, results in [('Paragraph', paragraph_results),
                                       ('Methodological', method_results)]:
            pipeline_scores = []
            for alg in ['kmeans', 'hdbscan_reassigned']:
                if 'clustering' in results and alg in results['clustering']:
                    score = results['clustering'][alg].get('metrics', {}).get('silhouette', -1)
                    pipeline_scores.append(score)
                else:
                    pipeline_scores.append(0)
            silhouette_data.append(pipeline_scores)
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, silhouette_data[0], width, label='Paragraph', color='#2E86AB')
        bars2 = ax1.bar(x + width/2, silhouette_data[1], width, label='Methodological', color='#A23B72')
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Clustering Quality Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
        
        # Number of clusters comparison
        cluster_counts = []
        for pipeline_name, results in [('Paragraph', paragraph_results),
                                       ('Methodological', method_results)]:
            pipeline_counts = []
            for alg in ['kmeans', 'hdbscan_reassigned']:
                if 'clustering' in results and alg in results['clustering']:
                    n_clusters = results['clustering'][alg].get('metrics', {}).get('n_clusters', 0)
                    pipeline_counts.append(n_clusters)
                else:
                    pipeline_counts.append(0)
            cluster_counts.append(pipeline_counts)
        
        bars3 = ax2.bar(x - width/2, cluster_counts[0], width, label='Paragraph', color='#2E86AB')
        bars4 = ax2.bar(x + width/2, cluster_counts[1], width, label='Methodological', color='#A23B72')
        
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Count Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
        
        plt.tight_layout()
        fig_path = self.figures_dir / 'comparison_metrics.pdf'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Publication figure saved to: {fig_path}")
        return fig_path
    
    def generate_thesis_chapter_structure(self):
        """Generate suggested thesis chapter structure for methodology"""
        chapter_structure = """
# Suggested Thesis Chapter Structure: Clustering Methodology

## Chapter X: Comparative Analysis of Document Clustering Pipelines

### X.1 Introduction
- Motivation for comparing paragraph-level vs methodological clustering
- Research questions addressed

### X.2 Methodology

#### X.2.1 Data Preparation
- Document preprocessing pipeline
- Feature extraction methods
  - Paragraph-level embeddings
  - Methodological summary embeddings

#### X.2.2 Algorithm Selection Rationale
- **Why K-means:**
  - Baseline algorithm for benchmarking
  - Interpretable cluster centers
  - Computational efficiency
  - Well-understood theoretical properties
  
- **Why HDBSCAN:**
  - Automatic noise detection
  - No need for pre-specifying k
  - Handles varying density clusters
  - Hierarchical structure analysis

- **Why not other algorithms:**
  - Spectral clustering: Computational cost
  - GMM: Gaussian assumption violations
  - DBSCAN: Superseded by HDBSCAN

#### X.2.3 Dimensionality Reduction Strategy
- UMAP for structure preservation
- Comparison with t-SNE and PCA
- Parameter optimization

#### X.2.4 Evaluation Framework
- Internal validation metrics
  - Silhouette coefficient
  - Calinski-Harabasz index
  - Davies-Bouldin index
- External validation
  - Keyword coherence analysis
  - Manual expert evaluation
- Cross-pipeline agreement metrics
  - Adjusted Rand Index
  - Normalized Mutual Information

### X.3 Experimental Setup
- Implementation details
- Computational resources
- Reproducibility measures

### X.4 Results

#### X.4.1 Pipeline Performance Comparison
[Insert Table: tab:pipeline_comparison]
[Insert Figure: comparison_metrics.pdf]

#### X.4.2 Cross-Pipeline Agreement Analysis
[Insert Table: tab:agreement_metrics]

#### X.4.3 Noise Handling and Reassignment
[Insert Table: tab:noise_reassignment]

#### X.4.4 Qualitative Analysis
- Cluster interpretation
- Representative document analysis
- Thematic coherence assessment

### X.5 Discussion

#### X.5.1 Pipeline Trade-offs
- Paragraph-level: Fine-grained analysis vs computational cost
- Methodological: Efficiency vs granularity loss

#### X.5.2 Algorithm Performance
- K-means: Stable but assumes spherical clusters
- HDBSCAN: Better natural structure discovery

#### X.5.3 Practical Recommendations
- When to use each pipeline
- Parameter tuning guidelines

### X.6 Conclusions
- Key findings summary
- Contributions to methodology
- Future work directions

## Appendices

### A. Hyperparameter Settings
### B. Complete Evaluation Metrics
### C. Cluster Representative Examples
"""
        
        # Save chapter structure
        structure_file = self.analysis_dir / 'thesis_chapter_structure.md'
        with open(structure_file, 'w') as f:
            f.write(chapter_structure)
        
        print(f"Thesis chapter structure saved to: {structure_file}")
        return structure_file
    
    def create_reproducibility_package(self):
        """Create complete reproducibility package"""
        package_dir = self.analysis_dir / 'reproducibility_package'
        package_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_content = """
# Clustering Experiments Reproducibility Package

## Contents

1. **configs/**: All configuration files
2. **scripts/**: All Python scripts
3. **data/**: Sample data format specification
4. **results/**: Key results and checkpoints
5. **environment/**: Environment setup instructions

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate thesis_clustering

# 2. Prepare your data
# Ensure CSV with columns matching config specification

# 3. Run experiments
python thesis_master_runner.py --generate-config
python thesis_master_runner.py --config configs/thesis_config.yaml --run-all

# 4. Analyze results
python thesis_results_analyzer.py --results-dir thesis_results_[timestamp]
```

## Citation

If you use this code, please cite:
[Your thesis citation here]

## License

[Your license here]
"""
        
        readme_file = package_dir / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create environment.yml
        env_content = """
name: thesis_clustering
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.21
  - pandas=1.3
  - scikit-learn=1.0
  - matplotlib=3.4
  - seaborn=0.11
  - scipy=1.7
  - pyyaml=5.4
  - tqdm=4.62
  - pip
  - pip:
    - hdbscan==0.8.27
    - umap-learn==0.5.2
    - kneed==0.7.0
"""
        
        env_file = package_dir / 'environment.yml'
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"Reproducibility package created in: {package_dir}")
        return package_dir
    
    def analyze_cluster_stability(self, results_dir: Path) -> Dict:
        """Analyze clustering stability across runs"""
        stability_analysis = {
            'paragraph': {},
            'methodological': {}
        }
        
        for pipeline in ['paragraph', 'methodological']:
            pipeline_dir = results_dir / pipeline / 'checkpoints'
            
            if not pipeline_dir.exists():
                continue
            
            # Load clustering results
            clustering_file = pipeline_dir / 'clustering.pkl'
            if clustering_file.exists():
                with open(clustering_file, 'rb') as f:
                    data = pickle.load(f)
                    clustering_results = data['data']
                
                for algorithm in ['kmeans', 'hdbscan']:
                    if algorithm in clustering_results:
                        labels = clustering_results[algorithm]['labels']
                        
                        # Compute stability metrics
                        unique_labels = np.unique(labels[labels >= 0])
                        cluster_sizes = [np.sum(labels == l) for l in unique_labels]
                        
                        stability_analysis[pipeline][algorithm] = {
                            'n_clusters': len(unique_labels),
                            'size_variance': np.var(cluster_sizes),
                            'size_cv': np.std(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes else 0,
                            'min_size': min(cluster_sizes) if cluster_sizes else 0,
                            'max_size': max(cluster_sizes) if cluster_sizes else 0,
                            'size_distribution': cluster_sizes
                        }
        
        return stability_analysis
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report combining all results"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE THESIS ANALYSIS REPORT")
        print("="*80)
        
        # Load all available results
        paragraph_results = {}
        method_results = {}
        comparison_results = {}
        
        # Try to load pipeline results
        for pipeline in ['paragraph', 'methodological']:
            checkpoint_dir = self.results_dir / pipeline / 'checkpoints'
            if checkpoint_dir.exists():
                results = {}
                for checkpoint_file in checkpoint_dir.glob('*.pkl'):
                    with open(checkpoint_file, 'rb') as f:
                        data = pickle.load(f)
                        results[checkpoint_file.stem] = data['data']
                
                if pipeline == 'paragraph':
                    paragraph_results = results
                else:
                    method_results = results
        
        # Load comparison results
        comparison_file = self.results_dir / 'pipeline_comparison' / 'comparison_results.json'
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                comparison_results = json.load(f)
        
        # Generate all outputs
        print("\n1. Generating LaTeX tables...")
        self.generate_latex_tables(paragraph_results, method_results, comparison_results)
        
        print("\n2. Creating publication figures...")
        self.create_publication_figures(paragraph_results, method_results)
        
        print("\n3. Generating thesis chapter structure...")
        self.generate_thesis_chapter_structure()
        
        print("\n4. Creating reproducibility package...")
        self.create_reproducibility_package()
        
        print("\n5. Analyzing cluster stability...")
        stability = self.analyze_cluster_stability(self.results_dir)
        
        # Create master report
        report_lines = [
            "# Comprehensive Thesis Clustering Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary of Outputs",
            "",
            "### 1. LaTeX Tables",
            f"- Location: {self.latex_dir}",
            "- Tables: Pipeline comparison, Agreement metrics, Noise statistics",
            "",
            "### 2. Publication Figures", 
            f"- Location: {self.figures_dir}",
            "- Format: PDF (publication-ready)",
            "",
            "### 3. Thesis Chapter Structure",
            f"- Location: {self.analysis_dir}/thesis_chapter_structure.md",
            "",
            "### 4. Reproducibility Package",
            f"- Location: {self.analysis_dir}/reproducibility_package",
            "",
            "## Key Findings",
            ""
        ]
        
        # Add stability analysis
        if stability:
            report_lines.append("### Cluster Stability Analysis")
            for pipeline in ['paragraph', 'methodological']:
                if pipeline in stability and stability[pipeline]:
                    report_lines.append(f"\n**{pipeline.title()} Pipeline:**")
                    for alg, metrics in stability[pipeline].items():
                        report_lines.append(f"- {alg}: {metrics['n_clusters']} clusters, "
                                         f"CV={metrics['size_cv']:.3f}")
        
        report_lines.extend([
            "",
            "## Recommendations for Thesis",
            "",
            "1. Use the paragraph pipeline for fine-grained analysis",
            "2. Use the methodological pipeline for paper-level grouping",
            "3. Report both pipelines to show robustness",
            "4. Include stability analysis to demonstrate reliability",
            "5. Provide complete reproducibility package as supplementary material",
            ""
        ])
        
        # Save master report
        report_content = "\n".join(report_lines)
        report_file = self.analysis_dir / 'comprehensive_analysis_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\n✓ Comprehensive report saved to: {report_file}")
        print("\n" + "="*80)
        
        return report_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Thesis Results Analysis and Reporting")
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--generate-latex', action='store_true',
                       help='Generate LaTeX tables only')
    parser.add_argument('--generate-figures', action='store_true',
                       help='Generate publication figures only')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run complete analysis and generate all outputs')
    
    args = parser.parse_args()
    
    analyzer = ThesisResultsAnalyzer(Path(args.results_dir))
    
    if args.full_analysis or (not args.generate_latex and not args.generate_figures):
        # Run complete analysis
        analyzer.generate_comprehensive_report()
    else:
        # Run specific components
        if args.generate_latex:
            print("Generating LaTeX tables...")
            # Load results and generate tables
            # (Would need to load results first)
            
        if args.generate_figures:
            print("Generating publication figures...")
            # Load results and generate figures
            # (Would need to load results first)
    
    print("\n✓ Analysis completed successfully!")


if __name__ == "__main__":
    main()
