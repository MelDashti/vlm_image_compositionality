#!/bin/bash
#SBATCH --job-name=clip_gde_open
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/clip_baseline/clip_gde_open_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/clip_baseline/clip_gde_open_%j.err

echo "========================================"
echo "CLIP Baseline - GDE Open-World"
echo "Settings: tolerance=1e-5, max_iter=100"
echo "Expected: ~27% (from GDE paper)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /etc/profile.d/modules.sh
module load miniconda3/25.7
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate dino-gde

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

cd /home/s3767876/analysis/vlm_image_compositionality

python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'ViT-L-14' \
  --model_pretraining 'openai' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world

echo ""
echo "End time: $(date)"
echo "========================================"
