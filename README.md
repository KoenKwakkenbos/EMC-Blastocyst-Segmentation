# Blastocyst Segmentation (RD-U-Net)
*Authors: Koen Kwakkenbos, Effrosyni Chavli, Muhammad Arif, Esther Baart, Jifke Veenland (Erasmus MC, Rotterdam, The Netherlands)*

## ✍️ Research

A Deep Learning repository for the segmentation of human blastocysts using a **Residual Dilated U-Net (RD-U-Net)**. This project includes custom data loading, specific loss functions (Weighted BCE + Dice), and morphological post-processing. The development and usage of this model was described in **[Title Paper]**. 

This repository provides a streamlined framework for reproducing the training process and includes an inference script to apply the pre-trained model to your own datasets.

## 📂 Structure
```text
.
├── src/
│   ├── dataset.py       # DataGenerator with Albumentations
│   ├── loss.py          # Weighted BCE + Dice Loss
│   ├── model.py         # RD-U-Net Architecture
│   └── utils.py         # Post-processing (hole filling, etc.)
├── train.py             # Training entry point
├── inference.py         # Inference entry point
└── requirements.txt
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/YourUsername/blastocyst-segmentation.git](https://github.com/YourUsername/blastocyst-segmentation.git)
cd blastocyst-segmentation
pip install -r requirements.txt

```

### 2. Verify Installation

Run the test script to ensure TensorFlow is working and the model can be built:

```bash
python test_installation.py

```

## 🧠 Usage

### Inference (Using Pre-trained Weights)

To generate segmentation masks for a folder of images:

```bash
python inference.py \\
    --img_dir "./data/test_images" \\
    --output_dir "./results" \\
    --model_path "./weights/best_model.h5" \\
    --postprocess

```

* `--postprocess`: Optional flag to apply morphological cleaning (filling holes, removing small artifacts).

### Training a New Model

Organize your data into two folders: one for images (`.jpg/.png`) and one for masks (`.tif`/`.png`).

```bash
python train.py \\
    --img_dir "./data/train_images" \\
    --mask_dir "./data/train_masks" \\
    --output_dir "./weights" \\
    --epochs 100 \\
    --batch_size 8

```


## 📊 Pre-trained Models

You can download the pre-trained weights from the [Releases](https://www.google.com/search?q=https://github.com/YourUsername/blastocyst-segmentation/releases) page.

* `best_model.h5`: Trained on our in-house dataset, described in the paper.


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
