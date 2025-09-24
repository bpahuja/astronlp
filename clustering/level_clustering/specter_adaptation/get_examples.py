"""
Cluster Analysis Script - Retrieve Paragraphs and Abstracts from Good Scoring Clusters

This script processes cluster evaluations, retrieves associated paragraphs and papers,
and fetches corresponding abstracts for analysis.

Usage Examples:
    # Quick test with 2 clusters
    python cluster_analysis.py --test-mode --embeddings-dir /path/to/embeddings
    
    # Process top 5 clusters with score >= 5
    python cluster_analysis.py --max-clusters 5 --min-score 5 --embeddings-dir /path/to/embeddings
    
    # Process all clusters with score >= 4 (no limit)
    python cluster_analysis.py --max-clusters -1 --min-score 4 --embeddings-dir /path/to/embeddings
    
    # Custom output file
    python cluster_analysis.py --output my_results.json --embeddings-dir /path/to/embeddings

Arguments:
    --min-score: Minimum score for filtering clusters (default: 4)
    --max-clusters: Maximum number of clusters to process (default: 10, use -1 for no limit)
    --embeddings-dir: Path to SPECTER2 embeddings directory
    --output: Output file path (default: cluster_analysis_results.json)
    --sample-size: Number of sample results to print (default: 2)
    --test-mode: Run in test mode with only 2 clusters for quick verification
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import ast
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_specter2_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load SPECTER2 embeddings and metadata from chunked format"""
    embeddings_dir = Path(embeddings_dir)
    
    # Load embedding info
    info_file = embeddings_dir / "embedding_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Embedding info not found: {info_file}")
    
    with info_file.open() as f:
        info = json.load(f)
    
    logger.info(f"Loading {info['total_paragraphs']} embeddings from {info['total_chunks']} chunks")
    
    # Get chunk directories
    chunk_dirs = sorted([d for d in embeddings_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("chunk_")])
    
    all_embeddings = []
    all_metadata = []
    
    for chunk_dir in chunk_dirs:
        emb_file = chunk_dir / "embeddings.npy"
        meta_file = chunk_dir / "metadata.csv"
        
        if emb_file.exists() and meta_file.exists():
            embeddings = np.load(emb_file).astype(np.float32)
            # Read metadata with paper_id as string type
            metadata = pd.read_csv(meta_file, dtype={'paper_id': str, 'para_id': str})
            
            all_embeddings.append(embeddings)
            all_metadata.append(metadata)
    
    # Combine
    X = np.vstack(all_embeddings)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    # Ensure paper_id and para_id are strings
    metadata_df['paper_id'] = metadata_df['paper_id'].astype(str)
    metadata_df['para_id'] = metadata_df['para_id'].astype(str)
    
    logger.info(f"Loaded {X.shape[0]} embeddings with dimension {X.shape[1]}")
    
    return X, metadata_df


def load_cluster_evaluations(jsonl_path: str) -> pd.DataFrame:
    """Load cluster evaluation data from JSONL file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} cluster evaluations")
    return df


def load_abstracts(abstracts_path: str) -> Dict[str, str]:
    """Load abstracts from JSONL file into a dictionary"""
    abstracts = {}
    with open(abstracts_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Store with the original paper_id format (with /)
            abstracts[entry['paper_id']] = entry['abstract']
    
    logger.info(f"Loaded {len(abstracts)} abstracts")
    return abstracts


def get_good_clusters(df: pd.DataFrame, min_score: int = 4, max_clusters: int = None) -> pd.DataFrame:
    """Filter clusters with good scores and optionally limit the number"""
    good_clusters = df[df['score'] >= min_score].copy()
    logger.info(f"Found {len(good_clusters)} clusters with score >= {min_score}")
    
    if max_clusters is not None and len(good_clusters) > max_clusters:
        # Sort by score descending to get the best clusters first
        good_clusters = good_clusters.sort_values('score', ascending=False).head(max_clusters)
        logger.info(f"Limited to top {max_clusters} clusters by score")
    
    return good_clusters


def parse_indices(indices_str: str) -> List[int]:
    """Parse the picked_indices string into a list of integers"""
    try:
        # The indices are stored as a string representation of a list
        return ast.literal_eval(indices_str)
    except:
        logger.error(f"Failed to parse indices: {indices_str}")
        return []


def retrieve_paragraphs_and_papers(good_clusters: pd.DataFrame, 
                                  metadata_df: pd.DataFrame,
                                  abstracts: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Retrieve paragraphs for good clusters and map to paper IDs and abstracts
    """
    results = []
    
    for _, row in good_clusters.iterrows():
        cluster_info = {
            'cluster_id': row['cluster_id'],
            'cluster_size': row['cluster_size'],
            'score': row['score'],
            'label': row['label'],
            'rationale': row['rationale'],
            'top_terms': row['top_terms'],
            'paragraphs': []
        }
        
        # Parse indices
        indices = parse_indices(row['picked_indices'])
        
        # Group by paper for this cluster
        paper_groups = {}
        
        for idx in indices:
            if idx < len(metadata_df):
                # Get paragraph metadata
                para_meta = metadata_df.iloc[idx]
                
                # Convert paper_id format from _ to / for abstract lookup
                paper_id_for_abstract = para_meta['paper_id'].replace('_', '/')
                
                # Group paragraphs by paper
                if para_meta['paper_id'] not in paper_groups:
                    paper_groups[para_meta['paper_id']] = {
                        'paper_id': para_meta['paper_id'],
                        'paper_id_formatted': paper_id_for_abstract,
                        'abstract': abstracts.get(paper_id_for_abstract, 'Abstract not found'),
                        'paragraphs': []
                    }
                
                paper_groups[para_meta['paper_id']]['paragraphs'].append({
                    'para_id': para_meta['para_id'],
                    'text': para_meta['text'],
                    'global_index': idx
                })
            else:
                logger.warning(f"Index {idx} out of bounds for metadata (size: {len(metadata_df)})")
        
        cluster_info['papers'] = list(paper_groups.values())
        results.append(cluster_info)
    
    return results


def analyze_label_distribution(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze the distribution of papers across different labels"""
    label_paper_map = {}
    
    for cluster in results:
        label = cluster['label']
        if label not in label_paper_map:
            label_paper_map[label] = set()
        
        for paper in cluster['papers']:
            label_paper_map[label].add(paper['paper_id'])
    
    # Convert to dataframe for better visualization
    label_stats = []
    for label, papers in label_paper_map.items():
        label_stats.append({
            'label': label,
            'num_papers': len(papers),
            'paper_ids': list(papers)[:5]  # Show first 5 paper IDs as example
        })
    
    return pd.DataFrame(label_stats).sort_values('num_papers', ascending=False)


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save the results to a JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_sample_results(results: List[Dict[str, Any]], num_samples: int = 2):
    """Print sample results for verification"""
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    for i, cluster in enumerate(results[:num_samples]):
        print(f"\nCluster {i+1}:")
        print(f"  Cluster ID: {cluster['cluster_id']}")
        print(f"  Label: {cluster['label']}")
        print(f"  Score: {cluster['score']}")
        print(f"  Top terms: {cluster['top_terms']}")
        print(f"  Number of papers: {len(cluster['papers'])}")
        
        if cluster['papers']:
            paper = cluster['papers'][0]  # Show first paper
            print(f"\n  Sample Paper:")
            print(f"    Paper ID: {paper['paper_id_formatted']}")
            print(f"    Number of paragraphs: {len(paper['paragraphs'])}")
            
            if paper['paragraphs']:
                para = paper['paragraphs'][0]
                print(f"\n    Sample Paragraph (ID: {para['para_id']}):")
                print(f"      {para['text'][:200]}...")
            
            print(f"\n    Abstract:")
            print(f"      {paper['abstract'][:300]}...")


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze cluster evaluations and retrieve papers')
    parser.add_argument('--min-score', type=int, default=4,
                       help='Minimum score for a good cluster (default: 4)')
    parser.add_argument('--max-clusters', type=int, default=10,
                       help='Maximum number of clusters to process (default: 10, use -1 for no limit)')
    parser.add_argument('--embeddings-dir', type=str, default="/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/embeddings_v1",
                       help='Path to SPECTER2 embeddings directory')
    parser.add_argument('--output', type=str, default="cluster_analysis_results.json",
                       help='Output file path (default: cluster_analysis_results.json)')
    parser.add_argument('--sample-size', type=int, default=2,
                       help='Number of sample results to print (default: 2)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with only 2 clusters for quick verification')
    
    args = parser.parse_args()
    
    # Configuration parameters
    MIN_SCORE = args.min_score
    MAX_CLUSTERS = 2 if args.test_mode else (None if args.max_clusters == -1 else args.max_clusters)
    
    # Define paths
    cluster_eval_path = "/vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/llm_v1/cluster_llm_evaluation_specter2.jsonl"
    embeddings_dir = Path(args.embeddings_dir)
    abstracts_path = "/vol/bitbucket/bp824/astro/data/abstracts/abstracts.jsonl"
    output_path = "test_" + args.output if args.test_mode else args.output
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    if args.test_mode:
        print("*** RUNNING IN TEST MODE ***")
    print(f"Minimum cluster score: {MIN_SCORE}")
    print(f"Maximum clusters to process: {MAX_CLUSTERS if MAX_CLUSTERS else 'No limit'}")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Output file: {output_path}")
    print("="*80 + "\n")
    
    try:
        # Step 1: Load cluster evaluations
        logger.info("Loading cluster evaluations...")
        cluster_df = load_cluster_evaluations(cluster_eval_path)
        
        # Step 2: Filter good clusters with limit
        logger.info(f"Filtering good clusters (score >= {MIN_SCORE}, max: {MAX_CLUSTERS})...")
        good_clusters = get_good_clusters(cluster_df, min_score=MIN_SCORE, max_clusters=MAX_CLUSTERS)
        
        if len(good_clusters) == 0:
            logger.warning(f"No clusters found with score >= {MIN_SCORE}")
            return
        
        # Step 3: Load embeddings and metadata
        logger.info("Loading embeddings and metadata...")
        X, metadata_df = load_specter2_embeddings(embeddings_dir)
        
        # Step 4: Load abstracts
        logger.info("Loading abstracts...")
        abstracts = load_abstracts(abstracts_path)
        
        # Step 5: Retrieve paragraphs and map to papers
        logger.info("Retrieving paragraphs and mapping to papers...")
        results = retrieve_paragraphs_and_papers(good_clusters, metadata_df, abstracts)
        
        # Step 6: Analyze label distribution
        logger.info("Analyzing label distribution...")
        label_stats = analyze_label_distribution(results)
        print("\n" + "="*80)
        print("LABEL DISTRIBUTION")
        print("="*80)
        print(label_stats.to_string())
        
        # Step 7: Print sample results
        print_sample_results(results, num_samples=min(args.sample_size, len(results)))
        
        # Step 8: Save results
        save_results(results, output_path)
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total clusters in file: {len(cluster_df)}")
        print(f"Clusters with score >= {MIN_SCORE}: {len(cluster_df[cluster_df['score'] >= MIN_SCORE])}")
        print(f"Clusters actually processed: {len(good_clusters)}")
        print(f"Total unique labels: {len(label_stats)}")
        if results:
            print(f"Average papers per cluster: {np.mean([len(c['papers']) for c in results]):.2f}")
            print(f"Total paragraphs retrieved: {sum(sum(len(p['paragraphs']) for p in c['papers']) for c in results)}")
        print(f"Results saved to: {output_path}")
        
        if args.test_mode:
            print("\n*** TEST MODE COMPLETE ***")
            print("If everything worked correctly, run without --test-mode to process all clusters")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please check that all file paths are correct")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()