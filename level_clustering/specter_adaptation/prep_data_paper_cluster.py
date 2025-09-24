#!/usr/bin/env python3
"""
Script to convert paragraph-level clustering results to paper-level vectors
suitable for the clustering experimentation framework.

Input: CSV with columns [para_id, paper_id, cluster]
Output: CSV with paper-level bag-of-clusters vectors
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

def create_paper_vectors_from_paragraphs(paragraph_csv_path: str, 
                                       output_csv_path: str,
                                       normalize: bool = True,
                                       exclude_noise: bool = False):
    """
    Convert paragraph clustering results to paper-level vectors.
    
    Args:
        paragraph_csv_path: Path to CSV with [para_id, paper_id, cluster] columns
        output_csv_path: Path for output CSV
        normalize: If True, normalize by total paragraphs (frequencies)
        exclude_noise: If True, exclude noise cluster (-1) from vectors
    """
    
    print(f"Loading paragraph results from: {paragraph_csv_path}")
    df = pd.read_csv(paragraph_csv_path)
    
    # Validate required columns
    required_cols = ['para_id', 'paper_id', 'cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df):,} paragraph labels")
    print(f"Unique papers: {df['paper_id'].nunique():,}")
    print(f"Unique clusters: {df['cluster'].nunique():,}")
    
    # Get all unique clusters
    all_clusters = sorted(df['cluster'].unique())
    
    # Optionally exclude noise cluster
    if exclude_noise and -1 in all_clusters:
        clusters_for_vectors = [c for c in all_clusters if c != -1]
        print(f"Excluding noise cluster (-1) from paper vectors")
    else:
        clusters_for_vectors = all_clusters
    
    print(f"Creating vectors with {len(clusters_for_vectors)} dimensions")
    
    # Group paragraphs by paper
    paper_data = []
    
    for paper_id, paper_paragraphs in df.groupby('paper_id'):
        # Count paragraphs in each cluster
        cluster_counts = Counter(paper_paragraphs['cluster'])
        total_paragraphs = len(paper_paragraphs)
        noise_count = cluster_counts.get(-1, 0)
        
        # Calculate normalization factor
        if exclude_noise:
            non_noise_paragraphs = total_paragraphs - noise_count
            normalization_factor = max(non_noise_paragraphs, 1)
        else:
            normalization_factor = total_paragraphs if normalize else 1
        
        # Create paper record
        paper_record = {
            'paper_id': paper_id,
            'total_paragraphs': total_paragraphs,
            'noise_paragraphs': noise_count,
            'noise_ratio': noise_count / total_paragraphs,
            'unique_clusters': len([c for c in cluster_counts.keys() if c != -1]),
            'non_noise_paragraphs': total_paragraphs - noise_count
        }
        
        # Add cluster features
        for cluster_id in clusters_for_vectors:
            count = cluster_counts.get(cluster_id, 0)
            if normalize:
                paper_record[f'cluster_{cluster_id}'] = count / normalization_factor
            else:
                paper_record[f'cluster_{cluster_id}'] = count
        
        paper_data.append(paper_record)
    
    # Create DataFrame
    result_df = pd.DataFrame(paper_data)
    
    # Save results
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"\nCreated paper vectors for {len(result_df):,} papers")
    print(f"Saved to: {output_csv_path}")
    print(f"Vector dimensionality: {len(clusters_for_vectors)}")
    
    # Show sample data
    print("\nSample data:")
    cluster_cols = [col for col in result_df.columns if col.startswith('cluster_')]
    print(result_df[['paper_id'] + cluster_cols[:5]].head())
    
    # Show statistics
    print(f"\nData statistics:")
    print(f"  Average paragraphs per paper: {result_df['total_paragraphs'].mean():.1f}")
    print(f"  Average noise ratio: {result_df['noise_ratio'].mean():.3f}")
    print(f"  Papers with 100% noise: {(result_df['noise_ratio'] == 1.0).sum()}")
    
    sparsity = (result_df[cluster_cols] == 0).mean().mean()
    print(f"  Feature sparsity: {sparsity:.3f}")
    
    return result_df

def validate_paper_vectors_csv(csv_path: str, vector_prefix: str = "cluster_"):
    """
    Validate that a CSV file is suitable for the clustering experiments.
    
    Args:
        csv_path: Path to the CSV file
        vector_prefix: Expected prefix for feature columns
    """
    print(f"Validating CSV file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded {len(df)} rows")
        
        # Check for required columns
        if 'paper_id' not in df.columns:
            print("✗ Missing 'paper_id' column")
            return False
        print("✓ Found 'paper_id' column")
        
        # Check for feature columns
        feature_cols = [col for col in df.columns if col.startswith(vector_prefix)]
        if len(feature_cols) == 0:
            print(f"✗ No feature columns found with prefix '{vector_prefix}'")
            return False
        print(f"✓ Found {len(feature_cols)} feature columns")
        
        # Check data types
        feature_data = df[feature_cols]
        if not feature_data.dtypes.apply(lambda x: x.kind in 'biufc').all():
            print("✗ Feature columns contain non-numeric data")
            return False
        print("✓ All feature columns are numeric")
        
        # Check for negative values
        if (feature_data < 0).any().any():
            print("⚠ Warning: Found negative values in features")
        else:
            print("✓ No negative values in features")
        
        # Check sparsity
        sparsity = (feature_data == 0).mean().mean()
        print(f"✓ Feature sparsity: {sparsity:.3f}")
        
        # Check value ranges
        max_val = feature_data.max().max()
        min_val = feature_data.min().min()
        print(f"✓ Value range: [{min_val:.3f}, {max_val:.3f}]")
        
        if max_val <= 1.0:
            print("✓ Data appears to be normalized (frequencies)")
        else:
            print("✓ Data appears to be raw counts")
        
        print("\n✅ CSV file validation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare paper-level data for clustering experiments")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert paragraph results to paper vectors')
    convert_parser.add_argument('--input', required=True, help='Input paragraph CSV file')
    convert_parser.add_argument('--output', required=True, help='Output paper vectors CSV file')
    convert_parser.add_argument('--normalize', action='store_true', default=True, 
                               help='Normalize by total paragraphs (default: True)')
    convert_parser.add_argument('--exclude_noise', action='store_true', default=False,
                               help='Exclude noise cluster from vectors')
    
    # Validate command  
    validate_parser = subparsers.add_parser('validate', help='Validate paper vectors CSV file')
    validate_parser.add_argument('--input', required=True, help='CSV file to validate')
    validate_parser.add_argument('--prefix', default='cluster_', help='Feature column prefix')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        create_paper_vectors_from_paragraphs(
            paragraph_csv_path=args.input,
            output_csv_path=args.output,
            normalize=args.normalize,
            exclude_noise=args.exclude_noise
        )
    elif args.command == 'validate':
        validate_paper_vectors_csv(args.input, args.prefix)
    else:
        parser.print_help()

# Example usage:
"""
# Convert paragraph results to paper vectors
python data_preparation.py convert --input labels_paragraph.csv --output paper_clusters.csv --normalize

# Validate the resulting CSV
python data_preparation.py validate --input paper_clusters.csv --prefix cluster_
"""