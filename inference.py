import os
import argparse
import numpy as np
import skimage.io
import cv2
from tensorflow.keras.models import load_model

from src.utils import postprocessing
from src.loss import weighted_bce_dice_loss

# Need to register custom loss to load model
custom_objects = {'weighted_bce_dice_loss': weighted_bce_dice_loss}

def main():
    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument("--img_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .h5 model file")
    parser.add_argument("--postprocess", action='store_true', help="Apply post-processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'overlays'), exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, custom_objects=custom_objects, compile=False)

    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"Found {len(image_files)} images.")

    for img_name in image_files:
        img_path = os.path.join(args.img_dir, img_name)
        
        # Read and resize if necessary (assuming model expects 800x800)
        img = skimage.io.imread(img_path, as_gray=True)
        
        # Prepare input (Add Batch and Channel dims)
        # Assuming images are already 800x800. If not, add resize logic here.
        img_input = np.expand_dims(img, axis=0) # Batch
        img_input = np.expand_dims(img_input, axis=3) # Channel
        
        # Predict
        pred = model.predict(img_input, verbose=0)
        pred_mask = (pred > 0.5).astype(np.float32)

        if args.postprocess:
            pred_mask = postprocessing(pred_mask)

        pred_mask = np.squeeze(pred_mask)

        # Save Mask
        mask_filename = img_name.rsplit('.', 1)[0] + '_mask.tif'
        save_path = os.path.join(args.output_dir, mask_filename)
        skimage.io.imsave(save_path, skimage.img_as_uint(pred_mask))

        # Save Overlay (Optional, for visual check)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Create a red mask (R=255, G=0, B=0)
        # We start with zeros
        color_mask = np.zeros_like(img_rgb)
        # Set the Red channel (channel 0 in skimage/RGB, but check if using opencv save it might be BGR)
        # skimage.io.imsave uses RGB, so channel 0 is Red.
        color_mask[:, :, 0] = pred_mask * 255 

        # Blend: Image + Mask
        # alpha=1.0 for image, beta=0.3 for mask to make it transparent
        overlay = cv2.addWeighted(img_rgb, 1.0, color_mask, 0.3, 0)
        
        # OPTIONAL: Draw a solid contour line for better visibility
        # contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2) # Draw 2px Red line

        skimage.io.imsave(os.path.join(args.output_dir, 'overlays', img_name), overlay)
    print("Inference complete.")

if __name__ == "__main__":
    main()
    