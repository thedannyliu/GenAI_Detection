# GenImage Dataset Setup Instructions

This directory contains instructions for acquiring and setting up the genimage dataset for AI-generated image detection.

## About GenImage

The **genimage** dataset is a million-scale benchmark for AI-generated image detection, containing pairs of real and AI-generated images across 1000 classes. It includes images from multiple generators:

- Midjourney
- Stable Diffusion v1.4 and v1.5
- ADM
- GLIDE
- Wukong
- VQDM
- BigGAN
- And others

## Dataset Structure

The complete genimage dataset follows this organization:

```
genimage/
├── <Generator1>/
│   ├── train/
│   │   ├── ai/
│   │   │   └── <class_folders>/  # 1000 ImageNet classes
│   │   └── nature/
│   │       └── <class_folders>/
│   └── val/
│       ├── ai/
│       │   └── <class_folders>/
│       └── nature/
│           └── <class_folders>/
├── <Generator2>/
│   └── ...
└── imagenet_ai/  # Combined dataset from all generators
    ├── train/
    │   ├── ai/
    │   │   └── <class_folders>/
    │   └── nature/
    │       └── <class_folders>/
    └── val/
        ├── ai/
        │   └── <class_folders>/
        └── nature/
            └── <class_folders>/
```

Our code primarily uses the `imagenet_ai` combined folder to access all generators' data together, but can also work with individual generator folders.

## Downloading the Dataset

The GenImage dataset can be downloaded from one of the following sources:

1. **Google Drive**: [GenImage Google Drive](https://drive.google.com/drive/folders/1NBJQvSCUMY7JBGV_FcGr5lL_Y9LJwe2P)
   
2. **Baidu Netdisk**: [GenImage Baidu Link](https://pan.baidu.com/s/1rlt1mUzG-mJ5Az4cMHHQxg?pwd=x9ah)

Note that the complete dataset is large (>250GB) and may require significant time to download.

## Setting Up the Dataset

1. **Download** the dataset archives from one of the links above.

2. **Extract** the archives to this directory (`data/`). You should have directories for each individual generator.

3. **Create the combined dataset structure** (if not already present in the downloaded files):

   ```bash
   # Example structure creation - adjust paths as needed
   mkdir -p genimage/imagenet_ai/train/ai
   mkdir -p genimage/imagenet_ai/train/nature
   mkdir -p genimage/imagenet_ai/val/ai
   mkdir -p genimage/imagenet_ai/val/nature
   
   # Copy or link files from individual generators
   # For example:
   # cp -r genimage/Midjourney/train/ai/* genimage/imagenet_ai/train/ai/
   # cp -r genimage/Stable_Diffusion_v1.5/train/ai/* genimage/imagenet_ai/train/ai/
   # etc.
   ```

   Note: The dataset authors may provide a script to create this combined structure - check their repository.

4. **Verify** your dataset structure matches the expected format:

   ```bash
   ls -l genimage/imagenet_ai/train/ai | wc -l    # Should show 1000 classes
   ls -l genimage/imagenet_ai/train/nature | wc -l  # Should show 1000 classes
   ```

## Using a Subset (Optional)

If you have limited disk space or need to run quick experiments, you can create a smaller subset:

```bash
# Create directories for the subset
mkdir -p genimage/subset/train/ai
mkdir -p genimage/subset/train/nature
mkdir -p genimage/subset/val/ai
mkdir -p genimage/subset/val/nature

# Copy a subset of classes (e.g., first 100 classes)
# This is just an example - adjust as needed
CLASSES=$(ls -l genimage/imagenet_ai/train/ai | head -100)
for CLASS in $CLASSES; do
    cp -r genimage/imagenet_ai/train/ai/$CLASS genimage/subset/train/ai/
    cp -r genimage/imagenet_ai/train/nature/$CLASS genimage/subset/train/nature/
    cp -r genimage/imagenet_ai/val/ai/$CLASS genimage/subset/val/ai/
    cp -r genimage/imagenet_ai/val/nature/$CLASS genimage/subset/val/nature/
done
```

Then update the config YAML to point to this subset.

## Configuration

Once the dataset is set up, you need to update the configuration to point to the correct location:

1. Edit `configs/default.yaml` or your experiment config file:

   ```yaml
   # Update the path to match your dataset location
   data:
     root_dir: "data/genimage/imagenet_ai"
     # If using a subset:
     # root_dir: "data/genimage/subset"
   ```

2. If you placed the dataset in a different location than the project directory, specify the full path:

   ```yaml
   data:
     root_dir: "/path/to/your/genimage/imagenet_ai"
   ```

## Dataset Splits

Our code uses the following splits:

- **Train**: All images under `train/` are used for model training
- **Validation**: All images under `val/` are used for validation during training and final evaluation
- **Cross-Generator**: For cross-generator experiments, we can use leave-one-out strategies (see the experiment plan)

## Questions and Issues

If you encounter issues with the dataset setup or have questions, please check the [official GenImage repository](https://github.com/GenImage-Dataset/GenImage) or open an issue in our project repository. 