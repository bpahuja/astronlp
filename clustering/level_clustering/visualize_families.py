import pandas as pd
import hdbscan
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import sys

# --- 1. CONFIGURATION ---
# --- You should tune these parameters for your specific dataset ---

# Input file from the previous step
INPUT_PROFILES_FILE = 'document_profiles.csv'

# HDBSCAN parameters
# This is now the most important parameter. It defines the smallest
# grouping of papers that can be considered a "family".
MIN_CLUSTER_SIZE = 5

# Dimensionality reduction parameters
# SVD components to reduce memory usage before clustering
SVD_COMPONENTS = 100  # Reduce from potentially thousands of features

# UMAP parameters for visualization
# n_neighbors controls the balance between local and global structure.
# min_dist controls how tightly points are packed.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Output file paths
OUTPUT_CSV_FILE = 'final_paper_families_hdbscan.csv'
OUTPUT_PLOT_2D_FILE = 'families_visualization_2d_hdbscan.png'
OUTPUT_PLOT_3D_FILE = 'families_visualization_3d_hdbscan.html'

def cluster_and_visualize_profiles(input_file: str):
    """
    Loads document profiles, clusters them into families using HDBSCAN,
    and generates 2D and 3D visualizations with memory-efficient processing.
    """
    # --- 2. LOAD DATA ---
    try:
        print(f"Loading document profiles from '{input_file}'...")
        profiles_df = pd.read_csv(input_file, index_col='paper_id')
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    if profiles_df.empty:
        print("Error: The input file is empty.")
        sys.exit(1)

    print(f"Loaded profiles for {len(profiles_df)} papers with {len(profiles_df.columns)} features.")

    # --- 3. MEMORY-EFFICIENT PREPROCESSING ---
    print("Preprocessing data with memory-efficient approach...")
    
    # Option 1: Use dense TF-IDF transformation (more memory efficient for clustering)
    print("Applying TF-IDF transformation...")
    try:
        transformer = TfidfTransformer(norm='l2', use_idf=True)
        # Convert to dense immediately to avoid sparse matrix memory issues
        scaled_profiles_sparse = transformer.fit_transform(profiles_df)
        scaled_profiles_dense = scaled_profiles_sparse.toarray()
        
        # Clear the sparse matrix to free memory
        del scaled_profiles_sparse
        
        print(f"TF-IDF transformation complete. Shape: {scaled_profiles_dense.shape}")
        
    except MemoryError:
        print("Memory error during TF-IDF. Trying alternative preprocessing...")
        # Alternative: Just use standard scaling
        scaler = StandardScaler()
        scaled_profiles_dense = scaler.fit_transform(profiles_df)
        print("Using StandardScaler instead of TF-IDF due to memory constraints.")
    
    # --- 4. DIMENSIONALITY REDUCTION TO SAVE MEMORY ---
    if scaled_profiles_dense.shape[1] > SVD_COMPONENTS:
        print(f"Reducing dimensionality from {scaled_profiles_dense.shape[1]} to {SVD_COMPONENTS} using SVD...")
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        scaled_profiles_reduced = svd.fit_transform(scaled_profiles_dense)
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
        
        # Clear the original dense matrix
        del scaled_profiles_dense
        scaled_profiles_final = scaled_profiles_reduced
    else:
        print("Data already has manageable dimensionality, skipping SVD...")
        scaled_profiles_final = scaled_profiles_dense

    # --- 5. CLUSTER DOCUMENTS INTO FAMILIES WITH HDBSCAN ---
    print(f"Clustering papers using HDBSCAN with min_cluster_size={MIN_CLUSTER_SIZE}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        algorithm='best'  # Let HDBSCAN choose the best algorithm
    )
    
    try:
        family_labels = clusterer.fit_predict(scaled_profiles_final)
        print("Clustering complete.")
    except MemoryError:
        print("Memory error during clustering. Trying with smaller min_cluster_size...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(MIN_CLUSTER_SIZE * 2, 10),  # Increase minimum cluster size
            metric='euclidean',
            cluster_selection_method='eom'
        )
        family_labels = clusterer.fit_predict(scaled_profiles_final)
        print(f"Clustering complete with adjusted min_cluster_size={max(MIN_CLUSTER_SIZE * 2, 10)}")

    # Create a results DataFrame to store all our new data
    results_df = profiles_df.copy()
    results_df['family_id'] = family_labels
    
    # Dynamically determine the number of clusters found (excluding noise)
    unique_labels = set(family_labels)
    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # --- 6. REDUCE DIMENSIONALITY WITH UMAP FOR VISUALIZATION ---
    print("Reducing dimensions for visualization using UMAP...")

    # 2D reduction
    try:
        umap_2d = umap.UMAP(
            n_components=2, 
            n_neighbors=min(UMAP_N_NEIGHBORS, len(profiles_df) - 1),
            min_dist=UMAP_MIN_DIST, 
            random_state=42,
            low_memory=True  # Use low memory mode
        )
        umap_2d_embeddings = umap_2d.fit_transform(scaled_profiles_final)
        results_df['umap_2d_x'] = umap_2d_embeddings[:, 0]
        results_df['umap_2d_y'] = umap_2d_embeddings[:, 1]
    except Exception as e:
        print(f"Error in 2D UMAP: {e}")
        print("Skipping 2D visualization...")
        results_df['umap_2d_x'] = np.nan
        results_df['umap_2d_y'] = np.nan

    # 3D reduction
    try:
        umap_3d = umap.UMAP(
            n_components=3, 
            n_neighbors=min(UMAP_N_NEIGHBORS, len(profiles_df) - 1),
            min_dist=UMAP_MIN_DIST, 
            random_state=42,
            low_memory=True
        )
        umap_3d_embeddings = umap_3d.fit_transform(scaled_profiles_final)
        results_df['umap_3d_x'] = umap_3d_embeddings[:, 0]
        results_df['umap_3d_y'] = umap_3d_embeddings[:, 1]
        results_df['umap_3d_z'] = umap_3d_embeddings[:, 2]
        print("Dimensionality reduction complete.")
    except Exception as e:
        print(f"Error in 3D UMAP: {e}")
        print("Skipping 3D visualization...")
        results_df['umap_3d_x'] = np.nan
        results_df['umap_3d_y'] = np.nan
        results_df['umap_3d_z'] = np.nan

    # --- 7. GENERATE AND SAVE VISUALIZATIONS ---
    print("Generating and saving visualizations...")

    # Generate and save 2D plot
    if not results_df['umap_2d_x'].isna().all():
        try:
            plt.figure(figsize=(14, 12))
            # Separate noise points for distinct plotting
            clustered_points = results_df[results_df['family_id'] != -1]
            noise_points = results_df[results_df['family_id'] == -1]
            
            # Plot clustered points with a vibrant palette
            if len(clustered_points) > 0:
                sns.scatterplot(
                    data=clustered_points,
                    x='umap_2d_x', y='umap_2d_y', hue='family_id',
                    palette=sns.color_palette("Spectral", n_colors=max(n_clusters_found, 1)),
                    s=50, alpha=0.9
                )
            
            # Plot noise points as grey crosses
            if len(noise_points) > 0:
                plt.scatter(
                    noise_points['umap_2d_x'], noise_points['umap_2d_y'],
                    marker='x', color='grey', s=30, alpha=0.5, label='Noise'
                )
            
            plt.title(f'2D UMAP Visualization of Discovered Methodological Families', fontsize=16)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.legend(title='Family ID (omitted)', loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOT_2D_FILE, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"Saved 2D visualization to '{OUTPUT_PLOT_2D_FILE}'")
        except Exception as e:
            print(f"Error generating 2D plot: {e}")

    # Generate and save 3D plot
    if not results_df['umap_3d_x'].isna().all():
        try:
            # Convert family_id to string to ensure discrete colors in Plotly
            results_df['family_id_str'] = results_df['family_id'].astype(str)
            
            fig_3d = px.scatter_3d(
                results_df,
                x='umap_3d_x', y='umap_3d_y', z='umap_3d_z',
                color='family_id_str',
                title=f'3D UMAP Visualization of Discovered Methodological Families',
                hover_name=results_df.index,  # Show paper_id on hover
                labels={'color': 'Family ID'}
            )
            fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
            fig_3d.write_html(OUTPUT_PLOT_3D_FILE)
            print(f"Saved interactive 3D visualization to '{OUTPUT_PLOT_3D_FILE}'")
        except Exception as e:
            print(f"Error generating 3D plot: {e}")

    # --- 8. SAVE FINAL RESULTS ---
    results_df.to_csv(OUTPUT_CSV_FILE)
    print(f"\nFinal results with family assignments saved to '{OUTPUT_CSV_FILE}'")
    
    # --- 9. Print Summary Statistics ---
    num_noise_points = (results_df['family_id'] == -1).sum()
    
    print("\n--- Clustering Summary ---")
    print(f"Number of families discovered: {n_clusters_found}")
    print(f"Number of papers classified as noise: {num_noise_points} ({num_noise_points/len(results_df):.2%})")
    
    if n_clusters_found > 0:
        print("\n--- Family Sizes (excluding noise) ---")
        family_sizes = results_df[results_df['family_id'] != -1]['family_id'].value_counts().sort_index()
        print(family_sizes)
    print("------------------------------------\n")


if __name__ == "__main__":
    cluster_and_visualize_profiles(INPUT_PROFILES_FILE)