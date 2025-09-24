import pandas as pd
import sys

# --- 1. CONFIGURATION ---
# The input file from your paragraph clustering step.
# It must contain 'paper_id' and 'cluster_id' columns.
INPUT_CLUSTERS_FILE = 'clustered_paragraphs.csv'

# The output file where the document profile vectors will be saved.
OUTPUT_PROFILES_FILE = 'document_profiles.csv'


def analyze_clusters_and_create_profiles(input_file: str, output_file: str):
    """
    Calculates the percentage of isolate papers and creates document profile vectors.

    Args:
        input_file (str): Path to the clustered paragraphs CSV file.
        output_file (str): Path to save the document profile vectors CSV.
    """
    # --- 2. LOAD DATA ---
    try:
        print(f"Loading data from '{input_file}'...")
        df = pd.read_csv(input_file)
        
        # Verify required columns exist
        required_columns = ['paper_id', 'cluster_id']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input CSV must contain the columns: {required_columns}")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    print("Data loaded successfully.")

    # --- 3. CALCULATE PERCENTAGE OF ISOLATE PAPERS ---
    print("\n--- Analyzing Isolate Papers ---")

    # Get the total number of unique papers
    total_papers = df['paper_id'].nunique()
    if total_papers == 0:
        print("No papers found in the input file.")
        return

    # Group by paper and check if all paragraphs in that paper are noise (cluster_id == -1)
    # The .all() method returns True if all values in the group meet the condition.
    isolate_check = df.groupby('paper_id')['cluster_id'].apply(lambda x: (x == -1).all())
    
    # Count the number of papers where the check was True
    num_isolate_papers = isolate_check.sum()
    
    # Calculate the percentage
    percentage_isolates = (num_isolate_papers / total_papers) * 100 if total_papers > 0 else 0

    print(f"Total number of unique papers: {total_papers}")
    print(f"Number of 'isolate' papers (all paragraphs are noise): {num_isolate_papers}")
    print(f"Percentage of isolate papers: {percentage_isolates:.2f}%")

    # --- 4. CREATE DOCUMENT PROFILE VECTORS ---
    print("\n--- Creating Document Profile Vectors ---")

    # Use pd.crosstab to create a frequency matrix of papers vs. cluster IDs.
    # This is the most direct way to create "bag-of-methods" vectors.
    # It will automatically include the -1 column if noise points exist.
    doc_profiles = pd.crosstab(df['paper_id'], df['cluster_id'])
    
    # Rename the -1 column for clarity
    if -1 in doc_profiles.columns:
        doc_profiles = doc_profiles.rename(columns={-1: 'noise_count'})

    print(f"Created profile vectors for {len(doc_profiles)} papers.")
    print(f"The profile vectors have {len(doc_profiles.columns)} features (snippet clusters + noise).")
    print("A preview of the document profiles:")
    print(doc_profiles.head())

    # --- 5. SAVE THE PROFILES ---
    doc_profiles.to_csv(output_file)
    print(f"\nDocument profiles successfully saved to '{output_file}'")


if __name__ == "__main__":
    analyze_clusters_and_create_profiles(INPUT_CLUSTERS_FILE, OUTPUT_PROFILES_FILE)