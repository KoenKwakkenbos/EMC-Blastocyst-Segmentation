import os
import sys
import numpy as np
import tensorflow as tf

# Attempt to import from local src
try:
    from src.model import build_rd_unet
    from src.loss import weighted_bce_dice_loss
    print("[+] Successfully imported local modules.")
except ImportError as e:
    print(f"[-] Error importing modules: {e}")
    print("    Make sure you are running this script from the root of the repository.")
    sys.exit(1)

def main():
    print("="*50)
    print("       BLASTOCYST SEGMENTATION: INSTALL TEST")
    print("="*50)

    # 1. Check System
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[+] GPU Detected: {gpus}")
    else:
        print("[!] No GPU detected. Running on CPU (Inference might be slow).")

    # 2. Test Model Construction
    print("\n[+] Building RD-U-Net model...")
    try:
        model = build_rd_unet(input_shape=(800, 800, 1), print_summary=False)
        print("    Model built successfully.")
    except Exception as e:
        print(f"[-] Model build failed: {e}")
        sys.exit(1)

    # 3. Test Dummy Inference
    print("\n[+] Running dummy inference (800x800x1 random noise)...")
    try:
        # Create a random image (Batch Size 1, Height 800, Width 800, Channels 1)
        dummy_input = np.random.rand(1, 800, 800, 1).astype(np.float32)
        
        # Run prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Check output shape
        if prediction.shape == (1, 800, 800, 1):
            print(f"    Prediction successful. Output shape: {prediction.shape}")
        else:
            print(f"[-] Prediction output shape mismatch: {prediction.shape}")
            sys.exit(1)

    except Exception as e:
        print(f"[-] Inference failed: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("SUCCESS: Environment is ready for training and inference.")
    print("="*50)

if __name__ == "__main__":
    main()
