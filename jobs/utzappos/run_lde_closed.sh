#!/bin/bash
#SBATCH --job-name=utzap_lde_closed
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/utzappos/lde_closed_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/utzappos/lde_closed_%j.err

echo "========================================"
echo "UT-Zappos: LDE Closed-World"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

source /etc/profile.d/modules.sh
module load miniconda3/25.7
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate dino-gde

cd /home/s3767876/analysis/vlm_image_compositionality

python classification.py \
  --dataset "ut-zappos" \
  --model_architecture "dinov2_vitl14" \
  --model_pretraining "facebook" \
  --experiment_name "LDE" \
  --modality_IW "image"

echo ""
echo "End time: $(date)"
