# DINOv2 Integration for Compositional Classification

This guide explains how to use **DINOv2** (a self-supervised vision model) with the GDE framework for **Experiment 1: Compositional Classification**.

## 🎯 Overview

**DINOv2** is a self-supervised vision transformer that produces high-quality image embeddings without language supervision. Unlike CLIP, DINOv2:
- Has **no text encoder** (vision-only)
- Is trained using **self-supervised learning** (no compositional labels)
- Produces **L2-normalized embeddings** on a hypersphere (compatible with spherical GDE)

This makes DINOv2 ideal for studying:
- **Pure visual compositionality** without language
- **Emergent compositional structure** in self-supervised models
- **Geometric properties** of visual representations

---

## 📦 Installation

DINOv2 is automatically downloaded via PyTorch Hub. No additional installation required!

The model will be downloaded on first use:
```bash
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
```

---

## 🚀 Quick Start

### Step 1: Extract DINOv2 Embeddings

Extract embeddings for your dataset (MIT-States or UT-Zappos):

```bash
# MIT-States with ViT-Large
python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_large

# UT-Zappos with ViT-Base
python -m datasets.compute_embeddings_dinov2 ut-zappos --model_size vit_base
```

**Available model sizes:**
- `vit_small` - 21M parameters, 384-dim embeddings
- `vit_base` - 86M parameters, 768-dim embeddings
- `vit_large` - 304M parameters, 1024-dim embeddings ⭐ **Recommended**
- `vit_giant` - 1.1B parameters, 1536-dim embeddings

**Time estimates:**
- MIT-States (~53k images): ~10-15 minutes on GPU
- UT-Zappos (~50k images): ~10-15 minutes on GPU

### Step 2: Run Compositional Classification

Run GDE with image-based ideal words:

```bash
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world
```

**Compare to baselines:**

```bash
# LDE (Linear Decomposable Embeddings)
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'LDE' \
  --modality_IW 'image' \
  --open_world
```

---

## 📊 Expected Results

### Performance Characteristics

| Method | Seen Acc | Unseen Acc | Harmonic Mean (H) | Notes |
|--------|----------|------------|-------------------|-------|
| **DINOv2 (no decomp)** | High | Low | Low | Poor generalization |
| **DINOv2 + LDE** | Medium | Medium | Medium | Linear decomposition |
| **DINOv2 + GDE** | Medium | **Higher** | **Higher** | Geometry-aware ✅ |
| **CLIP + GDE** | Highest | Highest | Highest | Language supervision |

**Key insight:** DINOv2 won't match CLIP's absolute performance (no compositional training), but **GDE should still outperform LDE and baselines**, demonstrating that compositional structure emerges even without language supervision.

---

## 🔬 Research Questions

Using DINOv2 with GDE, you can investigate:

1. **Does self-supervised learning discover compositional structure?**
   - Compare DINOv2 vs CLIP geometry
   - Are attribute/object primitives emergent or learned?

2. **How important is text supervision for compositionality?**
   - DINOv2 (no text) vs CLIP (with text) on same GDE framework
   - Performance gap = value of language alignment

3. **Can we transfer geometric structure across models?**
   - Train GDE on CLIP, apply to DINOv2 embeddings
   - Test generalization of compositional patterns

---

## ⚙️ Advanced Options

### Batch Size

Adjust batch size based on GPU memory:

```bash
# Smaller GPU (<8GB)
python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_base --batch_size 32

# Larger GPU (>16GB)
python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_large --batch_size 128
```

### Closed World vs Open World

```bash
# Closed world (only evaluate on train + test pairs)
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image'
  # No --open_world flag

# Open world (evaluate on all possible attribute-object combinations)
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world  # ← Add this flag
```

### Sample Size Control

Limit number of images per pair for faster experimentation:

```bash
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --n_images 10 \  # Use max 10 images per pair
  --open_world
```

---

## 🐛 Troubleshooting

### Error: "Text embeddings are not available for DINOv2 models"

**Cause:** You tried to use text-based ideal words with DINOv2.

**Solution:** DINOv2 is vision-only. Use `--modality_IW 'image'`:
```bash
python classification.py --modality_IW 'image' ...  # ✅ Correct
# Not: --modality_IW 'text'  # ❌ Wrong for DINOv2
```

### Error: "CUDA out of memory"

**Cause:** GPU doesn't have enough memory for batch size.

**Solution:** Reduce batch size:
```bash
python -m datasets.compute_embeddings_dinov2 mit-states --batch_size 32
```

Or use a smaller model:
```bash
python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_base
```

### Warning: "Embedding norms not close to 1.0"

**Cause:** DINOv2 embeddings should be L2-normalized.

**Solution:** This is automatically handled in `compute_embeddings_dinov2.py`. If you see this warning, check that normalization is applied:
```python
embs = embs / torch.norm(embs, p=2, dim=-1, keepdim=True)
```

---

## 📁 File Structure

After running DINOv2 extraction, your dataset directory will contain:

```
data/mit-states/
├── IMGemb_dinov2_vitl14_facebook.pt  # DINOv2 image embeddings
├── images/                            # Original images
└── compositional-split-natural/       # Train/val/test splits
```

**File format:**
```python
{
    'image_ids': [...],         # List of image paths
    'embeddings': Tensor,       # Shape: (N, dim)
    'pairs': [...]             # List of (attr, obj) tuples
}
```

---

## 🔍 Comparison with CLIP

| Feature | CLIP | DINOv2 |
|---------|------|--------|
| **Training** | Supervised (text-image pairs) | Self-supervised (images only) |
| **Text encoder** | ✅ Yes | ❌ No |
| **Image encoder** | ViT or ResNet | ViT |
| **Embeddings** | L2-normalized (sphere) | L2-normalized (sphere) |
| **Compositional training** | ✅ Explicit | ❌ Implicit |
| **GDE compatible** | ✅ Yes | ✅ Yes |
| **Ideal words modality** | Image or Text | Image only |
| **Zero-shot transfer** | Strong | Moderate |
| **Visual features** | Good | **Excellent** |

---

## 📚 References

1. **DINOv2 Paper:**
   - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
   - Oquab et al., 2023

2. **GDE Paper (this codebase):**
   - [Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models](https://arxiv.org/abs/2503.17142)
   - Berasi et al., CVPR 2025

3. **DINOv2 GitHub:**
   - https://github.com/facebookresearch/dinov2

---

## 💡 Tips

1. **Start with ViT-Large:** It offers the best performance-speed trade-off.

2. **Use image-based ideal words:** DINOv2 has no text encoder, so `--modality_IW 'image'` is required.

3. **Open world is harder:** Start with closed world to verify setup, then try open world.

4. **Compare to CLIP:** Run same experiments with CLIP to measure the value of language supervision.

5. **Visualize embeddings:** Use t-SNE or UMAP to visualize compositional structure in embedding space.

---

## 🆘 Support

For issues or questions:
1. Check the [main README](README.md) for general setup
2. See [classification.py](classification.py) for implementation details
3. Open an issue on GitHub

---

**Happy experimenting! 🚀**
