#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=summarization
#SBATCH --mem=64G
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=bp824
export PATH=/vol/bitbucket/bp824/astro/astroenv/bin/:$PATH
#PYTORCH_LIB_PATH=$(python -c "import os; import torch; print(os.path.dirname(torch.utils.cpp_extension.library_paths()[0]))")
#export LD_LIBRARY_PATH=$PYTORCH_LIB_PATH:$LD_LIBRARY_PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
#source activate
source /vol/bitbucket/bp824/astro/astroenv/bin/activate
source /vol/cuda/12.0.0/setup.sh

# DATA_PATH="/vol/bitbucket/bp824/astro/data/methodolodgy_labels/labeled_paragraphs.jsonl"
# OUTPUT_DIR="/vol/bitbucket/bp824/astro/data/methodolodgy_labels/output_astrobert"

python cluster_chunks_ii.py \
  --chunks_root /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/embeddings_v1_astrobert \
  --work_dir ./cluster_out_v8_astrobert \
  --backend umap_hdbscan \
  --umap_dim 50 \
  --n_neighbors 50 \
  --min_cluster_size 500 \
  --enhance_with_methods \
  --skip_normalization \
  --min_samples 50 --debug_memory --force_cpu
  # --sample_every 2
#   --max_chunks 10 \
# --num_samples 16 
