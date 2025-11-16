"""
DINOv2 Embedding Extraction for Compositional Classification

This script extracts DINOv2 embeddings for compositional datasets (MIT-States, UT-Zappos).
DINOv2 is a self-supervised vision model that produces L2-normalized embeddings lying on
a hypersphere, making it compatible with the GDE (Geodesically Decomposable Embeddings) framework.

Key differences from CLIP:
- No text encoder (image-only embeddings)
- Self-supervised training (no compositional labels)
- Strong visual features suitable for geometry analysis

Usage:
    python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_large
    python -m datasets.compute_embeddings_dinov2 ut-zappos --model_size vit_base
"""

import torch
import os
import argparse
from torchvision import transforms
from tqdm import tqdm

from datasets.read_datasets import DATASET_PATHS
from datasets.composition_dataset import CompositionDataset


def chunks(l, n):
    """Yield successive n-sized chunks from list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_dinov2_model(model_size='vit_large', device='cuda'):
    """
    Load pre-trained DINOv2 model from torch hub.

    Args:
        model_size (str): One of 'vit_small', 'vit_base', 'vit_large', 'vit_giant'
        device (str): Device to load model on

    Returns:
        model: DINOv2 model in eval mode
        model_name (str): Full model name for file naming
    """
    model_map = {
        'vit_small': 'dinov2_vits14',
        'vit_base': 'dinov2_vitb14',
        'vit_large': 'dinov2_vitl14',
        'vit_giant': 'dinov2_vitg14'
    }

    if model_size not in model_map:
        raise ValueError(f"Invalid model_size. Choose from {list(model_map.keys())}")

    model_name = model_map[model_size]
    print(f"Loading DINOv2 model: {model_name}...")

    # Load model from torch hub
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 model from torch hub. "
            f"Please check your internet connection and try again.\n"
            f"Error: {e}"
        )

    model = model.to(device)
    model.eval()

    print(f"OK Model loaded successfully on {device}")
    return model, model_name


def get_dinov2_transforms():
    """
    Get standard DINOv2 preprocessing transforms.

    DINOv2 uses ImageNet normalization and 224x224 input size.
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def compute_dinov2_embeddings(dataset_name, model_size='vit_large', batch_size=64):
    """
    Extract DINOv2 embeddings for all images in the specified dataset.

    Args:
        dataset_name (str): Dataset name ('mit-states', 'ut-zappos', 'waterbirds', 'celebA')
        model_size (str): DINOv2 model size
        batch_size (int): Batch size for embedding extraction

    Returns:
        output_path (str): Path where embeddings were saved
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model, model_name = get_dinov2_model(model_size, device)

    # Get preprocessing transforms
    preprocess = get_dinov2_transforms()

    # Load dataset
    try:
        dataset_path = DATASET_PATHS[dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {list(DATASET_PATHS.keys())}"
        )

    print(f"Loading dataset from: {dataset_path}")

    # Note: We pass transform=None and apply preprocessing manually in the loop
    # This gives us better error handling for individual images
    dataset = CompositionDataset(
        dataset_path,
        phase='all',  # Load all images (train + val + test)
        transform=None  # We'll apply transforms manually with error handling
    )

    print(f"Dataset: {dataset_name}")
    print(f"Total images: {len(dataset.data)}")

    # Check if embeddings already exist
    output_file = f'IMGemb_{model_name}_facebook.pt'
    output_path = os.path.join(dataset.root, output_file)

    if os.path.exists(output_path):
        print(f"\n⚠ WARNING: File {output_path} already exists!")
        response = input("Delete and recompute? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Keeping existing embeddings.")
            return output_path
        else:
            os.remove(output_path)
            print("OK Deleted existing file. Recomputing...")

    # Extract image paths and labels
    all_images, all_attrs, all_objs = zip(*dataset.data)

    print(f"\nExtracting embeddings...")
    print(f"Batch size: {batch_size}")
    print(f"Total images: {len(all_images)}")
    print(f"Batches: {len(all_images) // batch_size + 1}")

    # Extract embeddings
    image_embeddings = []
    successful_images = []
    successful_attrs = []
    successful_objs = []
    failed_images = []

    with torch.no_grad():
        img_idx = 0
        for img_chunk in tqdm(chunks(list(range(len(all_images))), batch_size),
                             total=len(all_images) // batch_size + 1,
                             desc='Computing DINOv2 embeddings'):
            # Load and preprocess images with error handling
            imgs = []
            batch_indices = []

            for idx in img_chunk:
                img_path = all_images[idx]
                try:
                    img = dataset.loader(img_path)  # Load PIL image
                    img = preprocess(img)  # Apply transforms
                    imgs.append(img)
                    batch_indices.append(idx)
                except Exception as e:
                    # Log failed image but continue processing
                    failed_images.append((img_path, str(e)))
                    # Skip this image
                    continue

            if len(imgs) == 0:
                # Entire batch failed, skip
                continue

            # Stack and move to device
            imgs = torch.stack(imgs).to(device)

            # Forward pass through DINOv2
            embs = model(imgs)

            # L2 normalize embeddings (project onto hypersphere)
            # This makes them compatible with spherical GDE
            embs = embs / torch.norm(embs, p=2, dim=-1, keepdim=True)

            image_embeddings.append(embs.cpu())

            # Track successful images and their labels
            for idx in batch_indices:
                successful_images.append(all_images[idx])
                successful_attrs.append(all_attrs[idx])
                successful_objs.append(all_objs[idx])

    # Report failed images if any
    if failed_images:
        print(f"\n⚠ WARNING: {len(failed_images)} images failed to load:")
        for img_path, error in failed_images[:10]:  # Show first 10
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    # Prepare final data with only successful images
    all_images = tuple(successful_images)
    all_pairs = list(zip(successful_attrs, successful_objs))

    # Concatenate all batches
    image_embeddings = torch.cat(image_embeddings, dim=0)

    print(f"\nOK Extracted {len(all_images)} embeddings")
    print(f"Embedding shape: {image_embeddings.shape}")
    print(f"Embedding dim: {image_embeddings.shape[1]}")

    # Verify L2 normalization
    norms = torch.norm(image_embeddings, p=2, dim=1)
    print(f"Embedding norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.6f}")

    # Save embeddings
    torch.save({
        'image_ids': all_images,
        'embeddings': image_embeddings,
        'pairs': all_pairs
    }, output_path)

    print(f"\nOK Saved embeddings to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract DINOv2 embeddings for compositional datasets'
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=['mit-states', 'ut-zappos', 'waterbirds', 'celebA'],
        help='Dataset name'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='vit_large',
        choices=['vit_small', 'vit_base', 'vit_large', 'vit_giant'],
        help='DINOv2 model size (default: vit_large)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for embedding extraction (default: 64)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DINOv2 EMBEDDING EXTRACTION")
    print("=" * 70)

    compute_dinov2_embeddings(
        dataset_name=args.dataset,
        model_size=args.model_size,
        batch_size=args.batch_size
    )

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Run compositional classification:")
    print(f"   python classification.py \\")
    print(f"     --dataset '{args.dataset}' \\")

    # Map model_size to correct architecture name
    model_map = {
        'vit_small': 'dinov2_vits14',
        'vit_base': 'dinov2_vitb14',
        'vit_large': 'dinov2_vitl14',
        'vit_giant': 'dinov2_vitg14'
    }
    model_arch = model_map[args.model_size]

    print(f"     --model_architecture '{model_arch}' \\")
    print(f"     --model_pretraining 'facebook' \\")
    print(f"     --experiment_name 'GDE' \\")
    print(f"     --modality_IW 'image' \\")
    print(f"     --open_world")
