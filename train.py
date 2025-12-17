import argparse
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from src.dataset import DataGenerator
from src.model import build_rd_unet
from src.loss import weighted_bce_dice_loss

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.1)

def main():
    parser = argparse.ArgumentParser(description='Train Blastocyst Segmentation Model')
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing training masks")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    all_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # --- METHOD 1: SIMPLE RANDOM SPLIT (Default for minimal example) ---
    # Good for single-frame-per-embryo datasets.
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # --- METHOD 2: EMBRYO-LEVEL SPLIT (Recommended for multi-frame data) ---
    # Use this if you have multiple frames per embryo (e.g., E001_1.jpg, E001_2.jpg)
    # to ensure the same embryo doesn't appear in both Train and Val.
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # 1. Parse Embryo IDs from filenames (e.g., "E001_frame1.jpg" -> "E001")
    # Adjust the parsing logic below to match your filename format
    embryo_ids = [f.split('_')[0] for f in all_files]
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(all_files, groups=embryo_ids))
    
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    
    print(f"Performed Embryo-level split. {len(np.unique(embryo_ids))} unique embryos.")
    """

    print(f"Training on {len(train_files)} images. Validating on {len(val_files)} images.")

    train_gen = DataGenerator(train_files, args.img_dir, args.mask_dir, 
                              batch_size=args.batch_size, augmentation=True)
    
    val_gen = DataGenerator(val_files, args.img_dir, args.mask_dir, 
                            batch_size=args.batch_size, augmentation=False, shuffle=False)

    # 2. Build Model
    model = build_rd_unet(input_shape=(800, 800, 1), normalization='min_max')
    
    model.compile(optimizer='adam', 
                  loss=weighted_bce_dice_loss, 
                  metrics=['accuracy'])

    # 3. Train
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_path = os.path.join(args.output_dir, f"best_model_{date_str}.h5")

    callbacks = [
        LearningRateScheduler(scheduler),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(train_gen, 
              validation_data=val_gen, 
              epochs=args.epochs, 
              callbacks=callbacks)

    print(f"Training complete. Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
