import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys

# --- 1. CONFIGURATION ---
# --- You should tune these parameters for your specific dataset ---

# Input file from the previous step
INPUT_PROFILES_FILE = 'document_profiles.csv'

# Number of methodological families you want to find
# This is the most important parameter to set.
N_CLUSTERS = 2000

# UMAP parameters for visualization
# n_neighbors controls the balance between local and global structure.
# min_dist controls how tightly points are packed.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Output file paths
OUTPUT_CSV_FILE = 'final_paper_families.csv'
OUTPUT_PLOT_2D_FILE = 'families_visualization_2d.png'
OUTPUT_PLOT_3D_FILE = 'families_visualization_3d.html'

def cluster_and_visualize_profiles(input_file: str):
    """
    Loads document profiles, clusters them into families, and generates
    2D and 3D visualizations.
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

    print(f"Loaded profiles for {len(profiles_df)} papers.")

    # --- 3. PREPROCESS DATA (TF-IDF TRANSFORMATION) ---
    print("Applying TF-IDF transformation to profile vectors...")
    # TfidfTransformer treats the counts as term frequencies (TF) and then
    # calculates the inverse document frequency (IDF), effectively down-weighting
    # common paragraph clusters (methods) that appear in many documents.
    transformer = TfidfTransformer(norm='l2', use_idf=True)
    scaled_profiles = transformer.fit_transform(profiles_df)

    # --- 4. CLUSTER DOCUMENTS INTO FAMILIES ---
    print(f"Clustering papers into {N_CLUSTERS} families using K-Means...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42, # For reproducible results
        n_init=10 # Run 10 times with different centroids to find the best fit
    )
    family_labels = kmeans.fit_predict(scaled_profiles)

    # Create a results DataFrame to store all our new data
    results_df = profiles_df.copy()
    results_df['family_id'] = family_labels
    print("Clustering complete.")

    # --- 5. REDUCE DIMENSIONALITY WITH UMAP ---
    print("Reducing dimensions for visualization using UMAP...")

    # 2D reduction
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=42
    )
    umap_2d_embeddings = umap_2d.fit_transform(scaled_profiles)
    results_df['umap_2d_x'] = umap_2d_embeddings[:, 0]
    results_df['umap_2d_y'] = umap_2d_embeddings[:, 1]

    # 3D reduction
    umap_3d = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=42
    )
    umap_3d_embeddings = umap_3d.fit_transform(scaled_profiles)
    results_df['umap_3d_x'] = umap_3d_embeddings[:, 0]
    results_df['umap_3d_y'] = umap_3d_embeddings[:, 1]
    results_df['umap_3d_z'] = umap_3d_embeddings[:, 2]

    print("Dimensionality reduction complete.")

    # --- 6. GENERATE AND SAVE VISUALIZATIONS ---
    print("Generating and saving visualizations...")

    # Generate and save 2D plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=results_df,
        x='umap_2d_x',
        y='umap_2d_y',
        hue='family_id',
        palette=sns.color_palette("Spectral", n_colors=N_CLUSTERS),
        s=50, # Marker size
        alpha=0.8
    )
    plt.title(f'2D UMAP Visualization of {N_CLUSTERS} Methodological Families', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Family ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_PLOT_2D_FILE, dpi=300)
    print(f"Saved 2D visualization to '{OUTPUT_PLOT_2D_FILE}'")

    # Generate and save 3D plot
    fig_3d = px.scatter_3d(
        results_df,
        x='umap_3d_x',
        y='umap_3d_y',
        z='umap_3d_z',
        color='family_id',
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f'3D UMAP Visualization of {N_CLUSTERS} Methodological Families',
        hover_name=results_df.index, # Show paper_id on hover
        labels={'color': 'Family ID'}
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
    fig_3d.write_html(OUTPUT_PLOT_3D_FILE)
    print(f"Saved interactive 3D visualization to '{OUTPUT_PLOT_3D_FILE}'")

    # --- 7. SAVE FINAL RESULTS ---
    results_df.to_csv(OUTPUT_CSV_FILE)
    print(f"\nFinal results with family assignments saved to '{OUTPUT_CSV_FILE}'")
    
    print("\n--- Summary of Families ---")
    print(results_df['family_id'].value_counts().sort_index())
    print("---------------------------\n")


if __name__ == "__main__":
    cluster_and_visualize_profiles(INPUT_PROFILES_FILE)