#!/bin/bash
#SBATCH --job-name=gde_open_gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gde_open_gpu_%j.out
#SBATCH --error=gde_open_gpu_%j.err

echo "========================================"
echo "GDE Open-World on GPU (Original Settings)"
echo "Settings: tolerance=1e-5, max_iter=10"
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
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world

echo ""
echo "End time: $(date)"
echo "========================================"
