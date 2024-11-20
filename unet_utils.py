## Implementing Single Image Segmentation Prediction in `unet_main.py`

#To perform segmentation overlay on a single RGB image using your trained U-Net model, follow the steps below. This involves modifying `unet_main.py` to accept a single image input, perform prediction, and overlay the segmentation mask on the original image.

### **Step 1: Update `load_model_and_predict` Function in `main.py`**

#Ensure that the `load_model_and_predict` function can handle both single images and batches. Modify it to accept an optional parameter for single image paths.

import tensorflow as tf
import numpy as np
import os
import logging
from config import CHECKPOINT_DIR
from model import build_unet  # Ensure this imports the U-Net model correctly
from utils import load_rgb_image, save_mask_overlay
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(img_dir: str, mask_dir: str, img_height: int = 256, img_width: int = 256, bands: int = 31) -> (np.ndarray, np.ndarray):
    """
    Loads HSI images and their corresponding masks from specified directories.

    Args:
        img_dir (str): Directory containing HSI images (multi-band TIFF).
        mask_dir (str): Directory containing mask images.
        img_height (int): Desired image height after resizing.
        img_width (int): Desired image width after resizing.
        bands (int): Number of spectral bands in HSI images.

    Returns:
        Tuple:
            X (np.ndarray): Array of HSI images with shape (num_samples, height, width, bands).
            Y (np.ndarray): Array of masks with shape (num_samples, height, width, 1).
    """
    try:
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tiff', '.tif'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.tiff', '.tif'))]
        
        logging.info(f"Found {len(img_files)} HSI images and {len(mask_files)} mask images.")

        # Create a mapping from image base name to mask file
        mask_dict = {os.path.splitext(f)[0].lower(): f for f in mask_files}
        
        X = []
        Y = []
        matched = 0
        skipped_no_mask = 0
        skipped_load_error = 0

        for img_file in img_files:
            img_base = os.path.splitext(img_file)[0].lower()
            mask_file = mask_dict.get(img_base)
            
            if not mask_file:
                logging.warning(f"No mask found for image '{img_file}'. Skipping.")
                skipped_no_mask += 1
                continue
            
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            try:
                # Load HSI image
                img = tiff.imread(img_path)
                
                # Verify shape
                if img.shape != (img_height, img_width, bands):
                    logging.warning(f"Resizing image '{img_file}' from {img.shape} to ({img_height}, {img_width}, {bands})")
                    img_resized = []
                    for b in range(img.shape[-1]):
                        band = Image.fromarray(img[:, :, b])
                        band_resized = band.resize((img_width, img_height), Image.BILINEAR)
                        img_resized.append(np.array(band_resized))
                    img = np.stack(img_resized, axis=-1)
                
                # Normalize HSI image
                img = img.astype('float32')
                img /= np.max(img) if np.max(img) != 0 else 1.0  # Avoid division by zero
                
                X.append(img)
                
                # Load and preprocess mask
                mask = Image.open(mask_path).convert('L')  # Convert to single channel
                mask = mask.resize((img_width, img_height), Image.NEAREST)
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=-1)  # Shape: (256, 256, 1)
                
                # Binarize mask: Assume mask pixels >0 belong to the class
                mask = np.where(mask > 0, 1, 0).astype('float32')
                
                Y.append(mask)
                matched += 1

            except Exception as e:
                logging.error(f"Failed to load image or mask for '{img_file}': {e}")
                skipped_load_error += 1
                continue

        X = np.array(X)
        Y = np.array(Y)
        logging.info(f"Loaded {matched} samples.")
        logging.info(f"Skipped {skipped_no_mask} images due to missing masks.")
        logging.info(f"Skipped {skipped_load_error} samples due to loading errors.")
        
        return X, Y

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return np.array([]), np.array([])
    
    
def load_model(checkpoint_path: str):
    """
    Load the trained U-Net model from the checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint directory.

    Returns:
        model (tf.keras.Model): Loaded U-Net model.
    """
    model = build_unet()
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        logging.info(f"Model restored from checkpoint: {latest_ckpt}")
    else:
        logging.error("No checkpoint found. Please check the checkpoint path.")
        raise FileNotFoundError("Checkpoint not found.")
    
    return model

def preprocess_image(image_path: str, target_size=(256, 256)) -> np.ndarray:
    """
    Load and preprocess the RGB image.

    Args:
        image_path (str): Path to the RGB image.
        target_size (tuple): Desired image size.

    Returns:
        image (np.ndarray): Preprocessed image array.
    """
    image = load_rgb_image(image_path, target_size)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)     # Add batch dimension
    return image

def predict_mask(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """
    Predict the segmentation mask using the U-Net model.

    Args:
        model (tf.keras.Model): Loaded U-Net model.
        image (np.ndarray): Preprocessed image array.

    Returns:
        mask (np.ndarray): Predicted mask.
    """
    prediction = model.predict(image)
    mask = (prediction > 0.5).astype(np.uint8)  # Binarize the mask
    mask = np.squeeze(mask, axis=0)            # Remove batch dimension
    return mask

def load_model_and_predict_single(image_path: str, checkpoint_path: str, output_path: str):
    """
    Load the model, predict the mask for a single image, and save the overlay.

    Args:
        image_path (str): Path to the input RGB image.
        checkpoint_path (str): Path to the checkpoint directory.
        output_path (str): Path to save the overlay image.

    Returns:
        None
    """
    # Load the model
    model = load_model(checkpoint_path)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    logging.info(f"Image loaded and preprocessed: {image_path}")
    
    # Predict the mask
    mask = predict_mask(model, image)
    logging.info("Mask prediction completed.")
    
    # Save the mask overlay
    save_mask_overlay(image_path, mask, output_path)
    logging.info(f"Overlay saved to: {output_path}")
