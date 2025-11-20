#!/bin/bash
#SBATCH --job-name=dinov1_gde_open
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/dinov1/gde_open_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/dinov1/gde_open_%j.err

echo "========================================"
echo "DINOv1 GDE Open-World - MIT-States"
echo "Model: dino_vitb16"
echo "Settings: tolerance=1e-5, max_iter=100"
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
cd ~/analysis/vlm_image_compositionality

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dino_vitb16' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world

echo ""
echo "End time: $(date)"
echo "========================================"
