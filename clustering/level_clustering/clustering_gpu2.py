import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
try:
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.decomposition import PCA as cuPCA
    from cuml.decomposition import IncrementalPCA as cuIncrementalPCA
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import cupy as cp
    import rmm
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with cuML/RAPIDS")
except ImportError as e:
    print(f"GPU libraries not available: {e}")
    print("Falling back to CPU versions...")
    import hdbscan
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    GPU_AVAILABLE = False

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Generator
import argparse
from collections import Counter
import pickle
import gc
import psutil
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import mmap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='clustering2.log')
logger = logging.getLogger(__name__)

def check_memory_usage():
    """Monitor memory usage (CPU and GPU)"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024**3)
    logger.info(f"Current CPU memory usage: {memory_gb:.2f} GB")
    
    if GPU_AVAILABLE:
        try:
            gpu_memory = cp.cuda.runtime.memGetInfo()
            gpu_free_gb = gpu_memory[0] / (1024**3)
            gpu_total_gb = gpu_memory[1] / (1024**3)
            gpu_used_gb = gpu_total_gb - gpu_free_gb
            logger.info(f"GPU memory usage: {gpu_used_gb:.2f}/{gpu_total_gb:.2f} GB")
            return memory_gb, gpu_used_gb
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return memory_gb, 0
    return memory_gb, 0

def setup_gpu_memory_pool():
    """Setup GPU memory pool for efficient memory management"""
    if GPU_AVAILABLE:
        try:
            # Initialize RMM memory pool with smaller initial size
            rmm.reinitialize(
                pool_allocator=True,
                managed_memory=False,
                initial_pool_size="1GB"  # Start smaller
            )
            logger.info("GPU memory pool initialized with 1GB")
        except Exception as e:
            logger.warning(f"Could not initialize GPU memory pool: {e}")

def get_array_info(file_path: str) -> Tuple[int, int, np.dtype]:
    """Get array shape and dtype without loading into memory"""
    with open(file_path, 'rb') as f:
        # Read numpy array header without loading data
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    return shape, dtype

def memory_mapped_array_chunks(file_path: str, chunk_size: int) -> Generator[np.ndarray, None, None]:
    """Generator that yields chunks of a memory-mapped array"""
    mmap_array = np.load(file_path, mmap_mode='r', allow_pickle=True)
    total_samples = mmap_array.shape[0]
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        # Load only this chunk into memory
        chunk = np.array(mmap_array[start_idx:end_idx], dtype=np.float32)
        yield chunk, start_idx, end_idx
        del chunk  # Explicit cleanup
        gc.collect()

class MemoryOptimizedParagraphClusterer:
    """
    Memory-optimized paragraph clustering system for large datasets.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embeddings_file = config['embeddings_file']
        self.df = None
        self.cluster_labels = None
        self.clusterer = None
        self.reduced_embeddings_file = None
        self.embedding_shape = None
        self.embedding_dtype = None
        self.chunk_size = min(config.get('chunk_size', 10000), 15000)  # Smaller default chunks
        self.gpu_available = GPU_AVAILABLE and config.get('use_gpu', True)
        
        if self.gpu_available:
            setup_gpu_memory_pool()
            logger.info("GPU acceleration enabled")
        else:
            logger.info("Using CPU-only processing")
        
        # Get embedding info without loading
        self.embedding_shape, self.embedding_dtype = get_array_info(self.embeddings_file)
        logger.info(f"Embedding shape: {self.embedding_shape}, dtype: {self.embedding_dtype}")
    
    def estimate_memory_requirements(self) -> Dict[str, float]:
        """Estimate memory requirements"""
        total_samples, embedding_dim = self.embedding_shape
        bytes_per_element = np.dtype(np.float32).itemsize  # Use float32
        
        # Estimate different memory requirements
        full_embeddings_gb = (total_samples * embedding_dim * bytes_per_element) / (1024**3)
        chunk_gb = (self.chunk_size * embedding_dim * bytes_per_element) / (1024**3)
        reduced_gb = (total_samples * self.config.get('target_dimensions', 100) * bytes_per_element) / (1024**3)
        
        estimates = {
            'full_embeddings': full_embeddings_gb,
            'chunk_size': chunk_gb,
            'reduced_embeddings': reduced_gb,
            'dataframe_approx': total_samples * 0.001  # Rough estimate for DataFrame
        }
        
        logger.info(f"Memory estimates (GB): {estimates}")
        return estimates
    
    # def load_metadata_only(self) -> pd.DataFrame:
    #     """Load only paragraph metadata without full text to save memory."""
    #     logger.info("Loading metadata only...")
        
    #     # Load ID mapping
    #     with open(self.config['id_to_index_file'], 'r') as f:
    #         id_to_index = json.load(f)
        
    #     # Create lightweight metadata
    #     metadata = []
    #     papers_dir = Path(self.config['papers_directory'])
        
    #     for txt_file in papers_dir.glob("*.txt"):
    #         paper_id = txt_file.stem
    #         try:
    #             # Get file size instead of reading content
    #             file_size = txt_file.stat().st_size
                
    #             # Count paragraphs without loading full content
    #             with open(txt_file, 'r', encoding='utf-8') as f:
    #                 para_count = 0
    #                 current_para = ""
    #                 for line in f:
    #                     if line.strip() == "":
    #                         if current_para.strip():
    #                             para_id = f"{paper_id}_para_{para_count:04d}"
    #                             if para_id in id_to_index:  # Only include if in embeddings
    #                                 metadata.append({
    #                                     'para_id': para_id,
    #                                     'paper_id': paper_id,
    #                                     'para_index': para_count,
    #                                     'embedding_index': id_to_index[para_id],
    #                                     'para_length': len(current_para)
    #                                 })
    #                             para_count += 1
    #                             current_para = ""
    #                     else:
    #                         current_para += line
                    
    #                 # Handle last paragraph
    #                 if current_para.strip():
    #                     para_id = f"{paper_id}_para_{para_count:04d}"
    #                     if para_id in id_to_index:
    #                         metadata.append({
    #                             'para_id': para_id,
    #                             'paper_id': paper_id,
    #                             'para_index': para_count,
    #                             'embedding_index': id_to_index[para_id],
    #                             'para_length': len(current_para)
    #                         })
                            
    #         except Exception as e:
    #             logger.warning(f"Error processing {txt_file}: {e}")
    #             continue
        
    #     df = pd.DataFrame(metadata)
    #     df = df.sort_values('embedding_index').reset_index(drop=True)
        
    #     logger.info(f"Loaded metadata for {len(df):,} paragraphs from {len(set(df['paper_id']))} papers")
    #     return df

    def load_metadata_only(self) -> pd.DataFrame:
        """Load only paragraph metadata from pre-generated mapping files to save memory."""
        logger.info("Loading metadata from pre-generated files...")
        
        # Path to the consolidated embeddings directory
        embeddings_dir = Path(self.config['embeddings_directory'])  # e.g., paragraph_embeddings_specter2_v4
        
        # Check for required metadata files
        id_to_idx_file = embeddings_dir / "paragraph_id_to_idx.json"
        paragraph_mapping_file = embeddings_dir / "paragraph_to_paper_mapping.json"
        
        if not id_to_idx_file.exists():
            raise FileNotFoundError(f"Paragraph ID to index mapping file not found: {id_to_idx_file}")
        
        if not paragraph_mapping_file.exists():
            raise FileNotFoundError(f"Paragraph to paper mapping file not found: {paragraph_mapping_file}")
        
        try:
            # Load paragraph to paper mapping (contains all metadata we need)
            logger.info("Loading paragraph to paper mapping...")
            with open(paragraph_mapping_file, 'r') as f:
                paragraph_mapping = json.load(f)
            
            # Convert to DataFrame directly from the mapping
            metadata = []
            for para_id, para_info in paragraph_mapping.items():
                metadata.append({
                    'para_id': para_id,
                    'paper_id': para_info['paper_id'],
                    'para_index': para_info['paragraph_index'],
                    'embedding_index': para_info['embedding_index'],
                    'para_length': para_info['text_length']
                })
            
            # Create DataFrame
            df = pd.DataFrame(metadata)
            
            # Sort by embedding index to maintain consistency with embeddings array
            df = df.sort_values('embedding_index').reset_index(drop=True)
            
            # Log statistics
            num_paragraphs = len(df)
            num_papers = df['paper_id'].nunique()
            avg_para_length = df['para_length'].mean()
            
            logger.info(f"Loaded metadata for {num_paragraphs:,} paragraphs from {num_papers:,} papers")
            logger.info(f"Average paragraph length: {avg_para_length:.1f} characters")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading metadata from pre-generated files: {e}")
            # Fallback to original method if pre-generated files are corrupted/incomplete
            logger.info("Falling back to original metadata loading method...")
            return self._load_metadata_fallback()

    def _load_metadata_fallback(self) -> pd.DataFrame:
        """Fallback method using original logic if pre-generated files are not available."""
        logger.info("Loading metadata using fallback method...")
        
        # Load ID mapping (if available)
        id_to_index_file = Path(self.config.get('id_to_index_file', ''))
        if not id_to_index_file.exists():
            logger.error(f"ID to index file not found: {id_to_index_file}")
            return pd.DataFrame()
        
        with open(id_to_index_file, 'r') as f:
            id_to_index = json.load(f)
        
        # Create lightweight metadata
        metadata = []
        papers_dir = Path(self.config['papers_directory'])
        
        if not papers_dir.exists():
            logger.error(f"Papers directory not found: {papers_dir}")
            return pd.DataFrame()
        
        txt_files = list(papers_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files to process")
        
        for txt_file in txt_files:
            paper_id = txt_file.stem
            try:
                # Count paragraphs without loading full content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    para_count = 0
                    current_para = ""
                    for line in f:
                        if line.strip() == "":
                            if current_para.strip():
                                para_id = f"{paper_id}_para_{para_count:04d}"
                                if para_id in id_to_index:  # Only include if in embeddings
                                    metadata.append({
                                        'para_id': para_id,
                                        'paper_id': paper_id,
                                        'para_index': para_count,
                                        'embedding_index': id_to_index[para_id],
                                        'para_length': len(current_para)
                                    })
                                para_count += 1
                                current_para = ""
                        else:
                            current_para += line
                    
                    # Handle last paragraph
                    if current_para.strip():
                        para_id = f"{paper_id}_para_{para_count:04d}"
                        if para_id in id_to_index:
                            metadata.append({
                                'para_id': para_id,
                                'paper_id': paper_id,
                                'para_index': para_count,
                                'embedding_index': id_to_index[para_id],
                                'para_length': len(current_para)
                            })
                            
            except Exception as e:
                logger.warning(f"Error processing {txt_file}: {e}")
                continue
        
        df = pd.DataFrame(metadata)
        if not df.empty:
            df = df.sort_values('embedding_index').reset_index(drop=True)
            logger.info(f"Loaded metadata for {len(df):,} paragraphs from {len(set(df['paper_id']))} papers")
        else:
            logger.warning("No metadata loaded!")
        
        return df
    
    def optimize_clustering_parameters_streaming(self) -> Dict:
        """
        Optimize parameters using streaming/sampling approach to minimize memory usage.
        """
        logger.info("Optimizing clustering parameters with streaming approach...")
        
        # Use smaller sample for optimization
        sample_size = min(8000, self.embedding_shape[0] // 20)
        logger.info(f"Using sample of {sample_size:,} points for parameter optimization")
        
        # Collect stratified sample
        total_samples = self.embedding_shape[0]
        step_size = max(1, total_samples // sample_size)
        sample_indices = np.arange(0, total_samples, step_size)[:sample_size]
        
        # Load sample using memory mapping
        mmap_embeddings = np.load(self.embeddings_file, mmap_mode='r', allow_pickle=True)
        sample_embeddings = np.array(mmap_embeddings[sample_indices], dtype=np.float32)
        del mmap_embeddings  # Release memory map
        
        # Quick PCA if high dimensional
        if sample_embeddings.shape[1] > 100:
            logger.info("Applying PCA for parameter optimization")
            if self.gpu_available:
                try:
                    sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
                    pca = cuPCA(n_components=min(50, sample_embeddings.shape[1]), random_state=42)
                    sample_embeddings = cp.asnumpy(pca.fit_transform(sample_gpu))
                    del sample_gpu
                    cp.cuda.runtime.deviceSynchronize()
                except Exception as e:
                    logger.warning(f"GPU PCA failed: {e}, using CPU")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(50, sample_embeddings.shape[1]), random_state=42)
                    sample_embeddings = pca.fit_transform(sample_embeddings).astype(np.float32)
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(50, sample_embeddings.shape[1]), random_state=42)
                sample_embeddings = pca.fit_transform(sample_embeddings).astype(np.float32)
        
        # Test parameters
        min_sizes = [30, 50, 80]  # Reduced options
        best_score = -1
        best_params = {}
        
        for min_size in min_sizes:
            try:
                if self.gpu_available:
                    sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
                    clusterer = cuHDBSCAN(
                        min_cluster_size=min_size,
                        min_samples=3,
                        cluster_selection_epsilon=0.0,
                        metric='euclidean'
                    )
                    labels = cp.asnumpy(clusterer.fit_predict(sample_gpu))
                    del sample_gpu
                    cp.cuda.runtime.deviceSynchronize()
                else:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_size,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        core_dist_n_jobs=1
                    )
                    labels = clusterer.fit_predict(sample_embeddings)
                
                if len(set(labels)) > 1 and -1 in labels:
                    mask = labels != -1
                    if mask.sum() > min_size * 2:
                        score = silhouette_score(sample_embeddings[mask], labels[mask])
                        
                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_ratio = (labels == -1).sum() / len(labels)
                        
                        logger.info(f"Min size {min_size}: {num_clusters} clusters, {noise_ratio:.2f} noise, silhouette: {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_cluster_size': min_size,
                                'silhouette_score': score,
                                'num_clusters': num_clusters,
                                'noise_ratio': noise_ratio
                            }
                
                del clusterer, labels
                gc.collect()
                if self.gpu_available:
                    cp.cuda.runtime.deviceSynchronize()
                            
            except Exception as e:
                logger.warning(f"Error with min_size {min_size}: {e}")
                continue
        
        # Clean up
        del sample_embeddings
        gc.collect()
        
        if best_params:
            logger.info(f"Best parameters: {best_params}")
            return best_params
        else:
            logger.warning("Could not find optimal parameters, using default")
            return {'min_cluster_size': max(50, self.config['min_cluster_size'])}
    
    def perform_streaming_dimensionality_reduction(self) -> str:
        """
        Perform dimensionality reduction in streaming fashion and save to disk.
        Returns path to reduced embeddings file.
        """
        target_dims = self.config.get('target_dimensions', 100)
        logger.info(f"Performing streaming dimensionality reduction: {self.embedding_shape[1]} -> {target_dims}")
        
        # Create output file for reduced embeddings
        reduced_file = self.embeddings_file.replace('.npy', '_reduced.npy')
        
        total_samples = self.embedding_shape[0]
        
        if self.gpu_available:
            try:
                logger.info("Using GPU incremental PCA")
                return self._gpu_streaming_pca(reduced_file, target_dims)
            except Exception as e:
                logger.warning(f"GPU streaming PCA failed: {e}, falling back to CPU")
        
        # CPU streaming PCA
        logger.info("Using CPU incremental PCA")
        batch_size = min(self.chunk_size, 5000)  # Smaller batches for PCA
        
        ipca = IncrementalPCA(n_components=target_dims, batch_size=batch_size)
        
        # Fit phase - process in chunks
        logger.info("Fitting incremental PCA...")
        processed_samples = 0
        
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(self.embeddings_file, batch_size):
            ipca.partial_fit(chunk)
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 10) == 0:
                logger.info(f"PCA fitting progress: {processed_samples:,}/{total_samples:,}")
                check_memory_usage()
            
            del chunk
            gc.collect()
        
        # Transform phase - process in chunks and save
        logger.info("Transforming embeddings...")
        
        # Pre-allocate output file
        reduced_shape = (total_samples, target_dims)
        reduced_mmap = np.memmap(reduced_file, dtype='float32', mode='w+', shape=reduced_shape)
        
        processed_samples = 0
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(self.embeddings_file, batch_size):
            transformed_chunk = ipca.transform(chunk).astype(np.float32)
            reduced_mmap[start_idx:end_idx] = transformed_chunk
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 10) == 0:
                logger.info(f"PCA transform progress: {processed_samples:,}/{total_samples:,}")
                check_memory_usage()
            
            del chunk, transformed_chunk
            gc.collect()
        
        # Ensure data is written to disk
        del reduced_mmap
        gc.collect()
        
        logger.info(f"Streaming dimensionality reduction complete. Explained variance ratio: {ipca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"Reduced embeddings saved to: {reduced_file}")
        
        return reduced_file
    
    def _gpu_streaming_pca(self, output_file: str, target_dims: int) -> str:
        """GPU streaming PCA implementation"""
        batch_size = min(self.chunk_size, 8000)  # Smaller GPU batches
        total_samples = self.embedding_shape[0]
        
        ipca = cuIncrementalPCA(n_components=target_dims, batch_size=batch_size)

        if os.path.exists(output_file.replace('.npy', '_converted.npy')):
            logger.info(f"Reduced file {output_file.replace('.npy', '_converted.npy')} already exists, skipping PCA")
            return output_file.replace('.npy', '_converted.npy')
        
        # Fit phase
        logger.info("Fitting GPU incremental PCA...")
        processed_samples = 0
        
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(self.embeddings_file, batch_size):
            chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
            ipca.partial_fit(chunk_gpu)
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 5) == 0:
                logger.info(f"GPU PCA fitting progress: {processed_samples:,}/{total_samples:,}")
            
            del chunk_gpu, chunk
            cp.cuda.runtime.deviceSynchronize()
            gc.collect()
        
        # Transform phase
        logger.info("Transforming with GPU PCA...")
        reduced_shape = (total_samples, target_dims)
        reduced_mmap = np.memmap(output_file, dtype='float32', mode='w+', shape=reduced_shape)
        
        processed_samples = 0
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(self.embeddings_file, batch_size):
            chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
            transformed_gpu = ipca.transform(chunk_gpu)
            transformed_chunk = cp.asnumpy(transformed_gpu).astype(np.float32)
            
            reduced_mmap[start_idx:end_idx] = transformed_chunk
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 5) == 0:
                logger.info(f"GPU PCA transform progress: {processed_samples:,}/{total_samples:,}")
            
            del chunk_gpu, transformed_gpu, transformed_chunk, chunk
            cp.cuda.runtime.deviceSynchronize()
            gc.collect()

        reduced_mmap.flush()
        
        del reduced_mmap
        gc.collect()

        from memmap_to_npy import convert_memmap_to_npy
        output_file = convert_memmap_to_npy(output_file, output_file.replace(".npy", "_converted.npy"),(total_samples, target_dims))
        
        logger.info("GPU streaming PCA complete")
        return output_file.replace(".npy", "_converted.npy")
    
    def perform_memory_efficient_clustering(self, embeddings_file: str, optimize_params: bool = True) -> np.ndarray:
        """
        Perform clustering with minimal memory usage.
        """
        logger.info("Starting memory-efficient clustering...")
        check_memory_usage()
        
        # Get optimal parameters first
        if optimize_params and not self.config.get('skip_optimization', False):
            optimal_params = self.optimize_clustering_parameters_streaming()
            min_cluster_size = optimal_params['min_cluster_size']
        else:
            min_cluster_size = max(30, self.config['min_cluster_size'])
        
        # Load embeddings with memory mapping first to check if we can fit in memory
        mmap_embeddings = np.load(embeddings_file, mmap_mode='r', allow_pickle=True)
        embedding_size_gb = mmap_embeddings.nbytes / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"Embeddings size: {embedding_size_gb:.2f} GB, Available memory: {available_memory_gb:.2f} GB")
        
        # Decide on clustering strategy based on available memory
        if embedding_size_gb * 1.5 < available_memory_gb:  # Conservative estimate
            logger.info("Sufficient memory available, loading embeddings for clustering")
            embeddings = np.array(mmap_embeddings, dtype=np.float32)
            del mmap_embeddings
            
            return self._direct_clustering(embeddings, min_cluster_size)
        else:
            logger.info("Insufficient memory for direct clustering, using chunked K-Means approach")
            del mmap_embeddings
            return self._chunked_clustering(embeddings_file, min_cluster_size)
    
    def _direct_clustering(self, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """Direct clustering when we have enough memory"""
        try:
            if self.gpu_available:
                logger.info("Attempting GPU HDBSCAN clustering...")
                
                # Check GPU memory
                embedding_size_gb = embeddings.nbytes / (1024**3)
                gpu_memory = cp.cuda.runtime.memGetInfo()
                gpu_free_gb = gpu_memory[0] / (1024**3)
                
                if embedding_size_gb * 2.5 < gpu_free_gb:
                    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
                    
                    self.clusterer = cuHDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=self.config.get('min_samples', 3),
                        cluster_selection_epsilon=0.0,
                        metric='euclidean'
                    )
                    
                    cluster_labels_gpu = self.clusterer.fit_predict(embeddings_gpu)
                    cluster_labels = cp.asnumpy(cluster_labels_gpu)
                    
                    del embeddings_gpu, cluster_labels_gpu
                    cp.cuda.runtime.deviceSynchronize()
                    
                    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    noise_points = (cluster_labels == -1).sum()
                    logger.info(f"GPU HDBSCAN complete: {num_clusters} clusters, {noise_points:,} noise points")
                    
                    return cluster_labels
                else:
                    raise MemoryError("Insufficient GPU memory")
            else:
                raise ImportError("GPU not available")
                
        except (MemoryError, ImportError, Exception) as e:
            logger.warning(f"Direct GPU clustering failed: {e}, using CPU HDBSCAN")
            
            try:
                clustering_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': self.config.get('min_samples', 3),
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom',
                    'algorithm': 'boruvka_kdtree',
                    'core_dist_n_jobs': 1,
                }
                
                self.clusterer = hdbscan.HDBSCAN(**clustering_params)
                cluster_labels = self.clusterer.fit_predict(embeddings)
                
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                noise_points = (cluster_labels == -1).sum()
                logger.info(f"CPU HDBSCAN complete: {num_clusters} clusters, {noise_points:,} noise points")
                
                return cluster_labels
                
            except MemoryError:
                logger.warning("CPU HDBSCAN also failed, falling back to chunked approach")
                # Save embeddings temporarily and use chunked approach
                temp_file = embeddings_file.replace('.npy', '_temp.npy')
                np.save(temp_file, embeddings)
                del embeddings
                gc.collect()
                
                result = self._chunked_clustering(temp_file, min_cluster_size)
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                return result
    
    def _chunked_clustering(self, embeddings_file: str, min_cluster_size: int) -> np.ndarray:
        """Fallback chunked clustering using MiniBatch K-Means"""
        logger.info("Using chunked MiniBatch K-Means clustering...")
        
        # Estimate reasonable number of clusters
        total_samples = self.embedding_shape[0]
        n_clusters = min(500, max(10, total_samples // (min_cluster_size * 3)))
        
        logger.info(f"Using {n_clusters} clusters for {total_samples:,} samples")
        
        if self.gpu_available:
            try:
                # Try GPU K-Means with streaming
                return self._gpu_chunked_kmeans(embeddings_file, n_clusters)
            except Exception as e:
                logger.warning(f"GPU chunked clustering failed: {e}, using CPU")
        
        # CPU MiniBatch K-Means
        batch_size = min(self.chunk_size, 5000)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,  # Reduce reassignment for memory efficiency
            max_no_improvement=20
        )
        
        # Fit incrementally
        logger.info("Fitting MiniBatch K-Means...")
        processed_samples = 0
        
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(embeddings_file, batch_size):
            kmeans.partial_fit(chunk)
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 20) == 0:
                logger.info(f"K-Means fitting progress: {processed_samples:,}/{total_samples:,}")
                check_memory_usage()
            
            del chunk
            gc.collect()
        
        # Predict labels in chunks
        logger.info("Predicting cluster labels...")
        cluster_labels = np.empty(total_samples, dtype=np.int32)
        
        processed_samples = 0
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(embeddings_file, batch_size):
            labels_chunk = kmeans.predict(chunk)
            cluster_labels[start_idx:end_idx] = labels_chunk
            processed_samples = end_idx
            
            if processed_samples % (batch_size * 20) == 0:
                logger.info(f"Prediction progress: {processed_samples:,}/{total_samples:,}")
            
            del chunk, labels_chunk
            gc.collect()
        
        self.clusterer = kmeans
        logger.info(f"MiniBatch K-Means complete: {n_clusters} clusters")
        
        return cluster_labels
    
    def _gpu_chunked_kmeans(self, embeddings_file: str, n_clusters: int) -> np.ndarray:
        """GPU chunked K-Means implementation"""
        logger.info("Using GPU chunked K-Means...")
        
        batch_size = min(self.chunk_size, 8000)
        total_samples = self.embedding_shape[0]
        
        # Initialize GPU K-Means
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            max_iter=100,
            tol=1e-4
        )
        
        # Collect a sample for initial fitting
        sample_size = min(50000, total_samples)
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
        
        mmap_embeddings = np.load(embeddings_file, mmap_mode='r', allow_pickle=True)
        sample_embeddings = np.array(mmap_embeddings[sample_indices], dtype=np.float32)
        del mmap_embeddings
        
        # Fit on sample
        sample_gpu = cp.asarray(sample_embeddings, dtype=cp.float32)
        kmeans.fit(sample_gpu)
        del sample_gpu, sample_embeddings
        cp.cuda.runtime.deviceSynchronize()
        gc.collect()
        
        # Predict labels in chunks
        logger.info("Predicting labels with GPU K-Means...")
        cluster_labels = np.empty(total_samples, dtype=np.int32)
        
        processed_samples = 0
        for chunk, start_idx, end_idx in memory_mapped_array_chunks(embeddings_file, batch_size):
            chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
            labels_gpu = kmeans.predict(chunk_gpu)
            cluster_labels[start_idx:end_idx] = cp.asnumpy(labels_gpu)
            
            processed_samples = end_idx
            if processed_samples % (batch_size * 10) == 0:
                logger.info(f"GPU prediction progress: {processed_samples:,}/{total_samples:,}")
            
            del chunk_gpu, labels_gpu, chunk
            cp.cuda.runtime.deviceSynchronize()
            gc.collect()
        
        self.clusterer = kmeans
        logger.info(f"GPU K-Means complete: {n_clusters} clusters")
        
        return cluster_labels
    
    def generate_labels_efficiently(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, str]:
        """Generate cluster labels efficiently for large datasets."""
        logger.info("Generating cluster labels efficiently...")
        
        # Use numpy for efficient counting
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        labels = {}
        for label, count in zip(unique_labels, counts):
            if label == -1:
                labels[int(label)] = "-1_Noise"
            else:
                labels[int(label)] = f"cluster_{label}_n{count}"
        
        return labels
    
    def save_results_streaming(self, df: pd.DataFrame, cluster_labels: np.ndarray, cluster_label_map: Dict):
        """Save results using streaming approach to minimize memory usage."""
        logger.info("Saving results with streaming approach...")
        
        output_file = self.config['output_file']
        chunk_size = min(self.config.get('save_chunk_size', 50000), 50000)  # Smaller chunks
        
        # Create header
        columns = ['para_id', 'paper_id', 'para_index', 'embedding_index', 'para_length', 'cluster_id', 'cluster_label']
        
        # Write header
        with open(output_file, 'w') as f:
            f.write(','.join(columns) + '\n')
        
        # Process and save in chunks
        total_rows = len(df)
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[i:end_idx].copy()
            
            # Add clustering results
            chunk_df['cluster_id'] = cluster_labels[i:end_idx]
            chunk_df['cluster_label'] = chunk_df['cluster_id'].map(cluster_label_map)
            
            # Append to file
            chunk_df.to_csv(output_file, mode='a', header=False, index=False)
            
            # Clean up chunk
            del chunk_df
            gc.collect()
            
            if i % (chunk_size * 5) == 0:
                logger.info(f"Saved rows {i:,} to {end_idx:,}")
        
        logger.info(f"Results saved to {output_file}")
        
        # Generate summary
        self._save_cluster_summary_streaming(cluster_labels, cluster_label_map)
    
    def _save_cluster_summary_streaming(self, cluster_labels: np.ndarray, cluster_label_map: Dict):
        """Save cluster summary using streaming approach."""
        summary_file = self.config['output_file'].replace('.csv', '_summary.txt')
        
        # Calculate statistics efficiently
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        num_clusters = len(unique_labels)
        total_points = len(cluster_labels)
        
        if -1 in unique_labels:
            num_clusters -= 1
            noise_idx = np.where(unique_labels == -1)[0]
            if len(noise_idx) > 0:
                noise_points = counts[noise_idx[0]]
                noise_percentage = noise_points / total_points * 100
            else:
                noise_points = 0
                noise_percentage = 0
        else:
            noise_points = 0
            noise_percentage = 0
        
        with open(summary_file, 'w') as f:
            f.write("PARAGRAPH CLUSTERING SUMMARY (Memory-Optimized)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total paragraphs: {total_points:,}\n")
            f.write(f"Number of clusters: {num_clusters:,}\n")
            f.write(f"Noise points: {noise_points:,} ({noise_percentage:.2f}%)\n")
            f.write(f"GPU acceleration: {'Enabled' if self.gpu_available else 'Disabled'}\n")
            f.write(f"Memory optimization: Enabled\n\n")
            
            # Top cluster sizes (sorted by count)
            f.write("TOP 30 CLUSTER SIZES\n")
            f.write("-" * 30 + "\n")
            
            # Sort by count (descending)
            sorted_indices = np.argsort(counts)[::-1]
            
            for i, idx in enumerate(sorted_indices[:30]):
                cluster_id = unique_labels[idx]
                if cluster_id != -1:  # Skip noise
                    count = counts[idx]
                    f.write(f"Cluster {cluster_id}: {count:,} paragraphs\n")
        
        logger.info(f"Cluster summary saved to {summary_file}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files to save disk space."""
        if hasattr(self, 'reduced_embeddings_file') and self.reduced_embeddings_file:
            try:
                if os.path.exists(self.reduced_embeddings_file):
                    os.remove(self.reduced_embeddings_file)
                    logger.info(f"Cleaned up temporary file: {self.reduced_embeddings_file}")
            except Exception as e:
                logger.warning(f"Could not clean up {self.reduced_embeddings_file}: {e}")
    
    def run_memory_optimized_pipeline(self):
        """Run the complete memory-optimized clustering pipeline."""
        logger.info("Starting memory-optimized clustering pipeline...")
        
        # Check memory requirements
        memory_estimates = self.estimate_memory_requirements()
        available_memory = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Available memory: {available_memory:.2f} GB")
        
        # Load lightweight metadata only
        self.df = self.load_metadata_only()
        check_memory_usage()
        
        # Determine if we need dimensionality reduction
        embeddings_file = self.embeddings_file
        
        if self.embedding_shape[1] > self.config.get('max_dimensions_direct', 150):
            logger.info("High dimensionality detected, applying dimensionality reduction")
            embeddings_file = self.perform_streaming_dimensionality_reduction()
            self.reduced_embeddings_file = embeddings_file
            gc.collect()
            check_memory_usage()
        
        # Perform clustering
        self.cluster_labels = self.perform_memory_efficient_clustering(embeddings_file)
        check_memory_usage()
        
        # Generate labels
        cluster_label_map = self.generate_labels_efficiently(self.df, self.cluster_labels)
        
        # Save results
        self.save_results_streaming(self.df, self.cluster_labels, cluster_label_map)
        
        # Clean up temporary files
        self.cleanup_temp_files()
        
        # Final cleanup
        gc.collect()
        if self.gpu_available:
            cp.cuda.runtime.deviceSynchronize()
        
        logger.info("Memory-optimized clustering pipeline completed successfully!")
        final_memory = check_memory_usage()
        
        return self.df, self.cluster_labels, cluster_label_map


def create_memory_optimized_config():
    """Create memory-optimized configuration."""
    return {
        # Input files
        'embeddings_file': 'embeddings.npy',
        'id_to_index_file': 'id_to_index.json',
        'papers_directory': 'papers/',
        
        # Output files
        'output_file': 'clustered_paragraphs.csv',
        
        # Memory optimization parameters
        'use_gpu': True,
        'chunk_size': 10000,  # Smaller chunks to reduce memory usage
        'save_chunk_size': 25000,  # Smaller save chunks
        'pca_batch_size': 5000,  # Smaller PCA batches
        'kmeans_batch_size': 8000,  # K-means batch size
        
        # Clustering parameters
        'min_cluster_size': 30,  # Smaller clusters for better memory usage
        'min_samples': 3,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        
        # Dimensionality reduction
        'target_dimensions': 100,
        'max_dimensions_direct': 150,  # Lower threshold
        
        # Pipeline optimizations
        'skip_optimization': False,
        'max_clusters_to_label': 200,
        'save_models': False,  # Don't save models to save memory and disk space
        'generate_visualizations': False,  # Disable for large datasets
        
        # Memory-specific settings
        'memory_efficient_mode': True,
        'cleanup_temp_files': True,
        'use_memory_mapping': True,
        'prefer_streaming': True,
    }


def check_system_requirements():
    """Check system requirements and provide recommendations."""
    logger.info("Checking system requirements...")
    
    # Check RAM
    memory_info = psutil.virtual_memory()
    total_ram_gb = memory_info.total / (1024**3)
    available_ram_gb = memory_info.available / (1024**3)
    
    logger.info(f"Total RAM: {total_ram_gb:.1f} GB")
    logger.info(f"Available RAM: {available_ram_gb:.1f} GB")
    
    if available_ram_gb < 8:
        logger.warning("Less than 8 GB RAM available. Consider closing other applications.")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_disk_gb = disk_usage.free / (1024**3)
    logger.info(f"Available disk space: {free_disk_gb:.1f} GB")
    
    if free_disk_gb < 10:
        logger.warning("Less than 10 GB disk space available. May need more for temporary files.")
    
    # Check GPU
    if GPU_AVAILABLE:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            for i in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode('utf-8')
                memory_gb = props['totalGlobalMem'] / (1024**3)
                logger.info(f"GPU {i}: {name} ({memory_gb:.1f} GB)")
        except Exception as e:
            logger.warning(f"GPU info unavailable: {e}")
    else:
        logger.info("No GPU acceleration available")


def main():
    """Main function with memory optimization."""
    parser = argparse.ArgumentParser(description='Memory-Optimized Paragraph Clustering')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--embeddings', type=str, default='embeddings.npy', help='Path to embeddings file')
    parser.add_argument('--id-map', type=str, default='id_to_index.json', help='Path to ID mapping file')
    parser.add_argument('--papers-dir', type=str, default='papers/', help='Directory containing paper text files')
    parser.add_argument('--output', type=str, default='clustered_paragraphs.csv', help='Output CSV file')
    parser.add_argument('--min-cluster-size', type=int, default=30, help='Minimum cluster size')
    parser.add_argument('--target-dims', type=int, default=100, help='Target dimensions after reduction')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip parameter optimization')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--pca-batch-size', type=int, default=5000, help='Batch size for PCA operations')
    parser.add_argument('--memory-check', action='store_true', help='Run system requirements check')
    
    args = parser.parse_args()
    
    # Check system requirements if requested
    if args.memory_check:
        check_system_requirements()
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_memory_optimized_config()
        
        # Override with command line arguments
        config.update({
            'embeddings_file': args.embeddings,
            'id_to_index_file': args.id_map,
            'papers_directory': args.papers_dir,
            'output_file': args.output,
            'min_cluster_size': args.min_cluster_size,
            'target_dimensions': args.target_dims,
            'chunk_size': args.chunk_size,
            'skip_optimization': args.skip_optimization,
            'use_gpu': not args.no_gpu,
            'pca_batch_size': args.pca_batch_size,
        })
    
    # Ensure input files exist
    if not os.path.exists(config['embeddings_file']):
        logger.error(f"Embeddings file not found: {config['embeddings_file']}")
        return
    
    if not os.path.exists(config['id_to_index_file']):
        logger.error(f"ID mapping file not found: {config['id_to_index_file']}")
        return
    
    if not os.path.exists(config['papers_directory']):
        logger.error(f"Papers directory not found: {config['papers_directory']}")
        return
    
    # Run system check
    check_system_requirements()
    
    # Run clustering
    try:
        clusterer = MemoryOptimizedParagraphClusterer(config)
        clusterer.run_memory_optimized_pipeline()
        
        logger.info("Clustering completed successfully!")
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise


def benchmark_memory_usage():
    """Benchmark memory usage of different approaches."""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARK")
    print("="*60)
    
    # Create sample data
    sample_sizes = [10000, 50000, 100000]
    dimensions = 384
    
    for sample_size in sample_sizes:
        print(f"\nTesting with {sample_size:,} samples, {dimensions} dimensions")
        
        # Create test data
        test_embeddings = np.random.randn(sample_size, dimensions).astype(np.float32)
        data_size_gb = test_embeddings.nbytes / (1024**3)
        print(f"Data size: {data_size_gb:.3f} GB")
        
        # Memory before
        initial_memory = psutil.virtual_memory().used / (1024**3)
        print(f"Initial memory: {initial_memory:.2f} GB")
        
        # Save and reload with memory mapping
        test_file = f'test_embeddings_{sample_size}.npy'
        np.save(test_file, test_embeddings)
        del test_embeddings
        gc.collect()
        
        # Test memory mapping
        mmap_embeddings = np.load(test_file, mmap_mode='r', allow_pickle=True)
        after_mmap = psutil.virtual_memory().used / (1024**3)
        print(f"After memory mapping: {after_mmap:.2f} GB (+{after_mmap - initial_memory:.3f} GB)")
        
        # Test loading chunk
        chunk = np.array(mmap_embeddings[:1000])
        after_chunk = psutil.virtual_memory().used / (1024**3)
        print(f"After loading 1K chunk: {after_chunk:.2f} GB (+{after_chunk - initial_memory:.3f} GB)")
        
        # Cleanup
        del mmap_embeddings, chunk
        os.remove(test_file)
        gc.collect()
        
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check for special commands
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark-memory":
        benchmark_memory_usage()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check-requirements":
        check_system_requirements()
    else:
        main()