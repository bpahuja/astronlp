#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=summarization
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

python cluster_paragraph_complete.py \
  --chunks_root /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/embeddings_v1_astrobert \
  --work_dir results/paragraph_pipeline2 \
  --backend ipca_hdbscan --ipca_dim 128 \
  --min_cluster_size 12 --min_samples 1 \
  --sweep_min_cluster_size 8,12,16,24 \
  --sweep_min_samples 1,3,5 \
  --seed_heatmap_runs 5 --seed_heatmap_shuffle --force_cpu
#   --paper_labels_csv results/paper_pipeline2/final_hdbscan_labels.csv \
#   --export_llm_staging --meta_text_columns paragraph_text \
#   --topk 5 --topk_eval_cap 2000
# --num_samples 16 
