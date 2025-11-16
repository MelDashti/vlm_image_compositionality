"""
Extract train_pairs.txt, val_pairs.txt, test_pairs.txt from metadata .t7 file.

This script is needed because the original compositional split download URL is broken.
The CAILA mirror provides metadata files but not the split txt files, so we extract
them from the metadata.

Usage: python extract_splits_from_metadata.py <dataset_name>
Example: python extract_splits_from_metadata.py mit-states
"""
import os
import sys
import torch
from pathlib import Path

def extract_splits(dataset_name):
    """Extract split .txt files from metadata .t7 file."""

    data_root = Path("data") / dataset_name
    metadata_file = data_root / "metadata_compositional-split-natural.t7"
    output_dir = data_root / "compositional-split-natural"

    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found!")
        return False

    print(f"Loading metadata from {metadata_file}...")
    data = torch.load(str(metadata_file), weights_only=False)

    # Collect pairs for each split
    train_pairs = set()
    val_pairs = set()
    test_pairs = set()

    for instance in data:
        attr = instance.get("attr")
        obj = instance.get("obj")
        settype = instance.get("set")

        # Skip invalid entries
        if attr == "NA" or obj == "NA" or settype == "NA":
            continue

        pair = (attr, obj)

        if settype == "train":
            train_pairs.add(pair)
        elif settype == "val":
            val_pairs.add(pair)
        elif settype == "test":
            test_pairs.add(pair)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Write split files
    def write_pairs(filename, pairs):
        filepath = output_dir / filename
        sorted_pairs = sorted(list(pairs))
        with open(filepath, "w") as f:
            for attr, obj in sorted_pairs:
                f.write(f"{attr} {obj}\n")
        print(f"OK Created {filepath} ({len(sorted_pairs)} pairs)")

    write_pairs("train_pairs.txt", train_pairs)
    write_pairs("val_pairs.txt", val_pairs)
    write_pairs("test_pairs.txt", test_pairs)

    print(f"\nOK Successfully extracted splits for {dataset_name}!")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_splits_from_metadata.py <dataset_name>")
        print("Example: python extract_splits_from_metadata.py mit-states")
        sys.exit(1)

    dataset_name = sys.argv[1]
    success = extract_splits(dataset_name)
    sys.exit(0 if success else 1)
