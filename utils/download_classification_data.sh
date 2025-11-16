#!/bin/sh

CURRENT_DIR=$(pwd)

mkdir -p data
cd data

# Download datasets and splits
wget -c http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip -O mitstates.zip
wget -c http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip -O utzap.zip

# Try to download compositional splits from original source (currently broken)
echo "Attempting to download compositional splits from original source..."
wget -c https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz -O compositional_split_natural.tar.gz

# Check if download succeeded, otherwise use CAILA mirror
if [ ! -f compositional_split_natural.tar.gz ] || [ ! -s compositional_split_natural.tar.gz ]; then
    echo "Original source failed. Downloading from CAILA mirror (Google Drive)..."
    wget -c "https://drive.usercontent.google.com/u/0/uc?id=1q7G7yYAvCE9j9fWVGpEwJwtJjOVL97Xf&export=download" -O compositional_split_natural.tar.gz
fi

# MIT-States
unzip -q mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
#rename "s/ /_/g" mit-states/images/*   ## Doesn't work everywhere
for file in mit-states/images/*; do mv "$file" "$(echo "$file" | sed 's/ /_/g')"; done

# UT-Zappos50k
unzip -q utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

# Extract compositional splits
echo "Extracting compositional splits..."
tar -zxvf compositional_split_natural.tar.gz

# Extract split .txt files from metadata if they're missing
# (CAILA mirror only contains metadata, not the split txt files)
cd $CURRENT_DIR
if [ ! -f data/mit-states/compositional-split-natural/train_pairs.txt ]; then
    echo "Split txt files missing. Extracting from metadata..."
    python3 extract_splits_from_metadata.py mit-states
    python3 extract_splits_from_metadata.py ut-zappos

    # Copy ut-zap50k metadata and splits to ut-zappos
    cp -r data/ut-zap50k/compositional-split-natural data/ut-zappos/
    cp data/ut-zap50k/metadata_compositional-split-natural.t7 data/ut-zappos/
fi

cd data
rm -r mitstates.zip utzap.zip compositional_split_natural.tar.gz

cd $CURRENT_DIR
python3 datasets/reorganize_utzap.py

mv data/ut-zap50k data/ut-zappos 2>/dev/null || true

echo "New dataset created: data/mit-states"
echo "New dataset created: data/ut-zappos"
