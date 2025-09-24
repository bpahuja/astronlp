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

# python astro_methods_bakeoff_2.py \
#   --pairs_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/pairs \
#   --out_root  ./bakeoff_out \
#   --bases allenai/specter2_base,adsabs/astroBERT \
#   --lexicon_json /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/lexicon_v2.json \
#   --epochs 6 --batch_size 128 --grad_accum 2 --lr 1e-4 \
#   --window_tokens 180 --stride 30 --max_seq_length 256 \
#   --paras_dir /vol/bitbucket/bp824/astro/data/methodology_dataset \
#   --cluster_sample 100 \
#   --umap_dim 50 --n_neighbors 150 --min_cluster_size 10 --min_samples 2
#   --mask_prob 0.5 --device auto --hf_cache /vol/bitbucket/bp824/hf_models

# python astro_methods_bakeoff_3.py \
#   --mode hparam_search \
#   --base_model adsabs/astroBERT \
#   --pairs_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/pairs \
#   --out_root ./search_astrobert \
#   --lexicon_json /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/lexicon_v2.json \
#   --search_trials 10 --search_epochs 4 \
#   --window_tokens 180 --stride 30 --max_seq_length 256 \
#   --hf_cache /vol/bitbucket/bp824/hf_models --device auto

python astro_methods_bakeoff_3.py \
  --pairs_dir /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/pairs \
  --out_root ./final_astrobert_pfeiff12_lr1e4 \
  --bases adsabs/astroBERT \
  --lexicon_json /vol/bitbucket/bp824/astro/level_clustering/specter_adaptation/lexicon_v2.json\
  --paras_dir /vol/bitbucket/bp824/astro/data/methodology_dataset --cluster_sample 200 \
  --umap_dim 50 --n_neighbors 180 --min_cluster_size 10 --min_samples 2 \
  --hf_cache /vol/bitbucket/bp824/hf_models --device auto
