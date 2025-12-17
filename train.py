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
    
    # Simple 80/20 Train/Val split
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

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
