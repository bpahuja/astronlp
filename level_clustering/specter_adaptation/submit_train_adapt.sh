#!/bin/bash
#SBATCH --gres=gpu:1
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
#source /vol/cuda/12.0.0/setup.sh

# DATA_PATH="/vol/bitbucket/bp824/astro/data/methodolodgy_labels/labeled_paragraphs.jsonl"
# OUTPUT_DIR="/vol/bitbucket/bp824/astro/data/methodolodgy_labels/output_astrobert"

# python train_adapter.py \
#   --pairs_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/pairs \
#   --out_dir  /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/astrobert_adapter_best_config \
#   --device auto \
#   --batch_size 64 --grad_accum 2 \
#   --epochs 8 --lr 1e-4 \
#   --window_tokens 180 --stride 30 --max_seq_length 256 \
#   --num_workers 4

# python eval_pairs.py --pairs_jsonl /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/pairs/pairs_test.jsonl --model_path /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/astrobert_adapter_best_config --base_model adsabs/astroBERT 


python embed.py \
    --paras_dir /vol/bitbucket/bp824/astro/data/methodology_dataset \
    --model_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/astrobert_adapter_best_config \
    --out_dir   ./embeddings_v1_astrobert \
    --base_model adsabs/astroBERT \
    --window_tokens 180 --stride 30 --max_seq_length 512 --device auto
    # --umap_dim  100 --min_cluster_size 10 --min_samples 2 \

# python llm_eval_para.py 

# --num_samples 16 
