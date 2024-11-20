## Implementing Single Image Segmentation Prediction in `unet_main.py`

#To perform segmentation overlay on a single RGB image using your trained U-Net model, follow the steps below. This involves modifying `unet_main.py` to accept a single image input, perform prediction, and overlay the segmentation mask on the original image.

### **Step 1: Update `load_model_and_predict` Function in `main.py`**

#Ensure that the `load_model_and_predict` function can handle both single images and batches. Modify it to accept an optional parameter for single image paths.

import tensorflow as tf
import numpy as np
import os
import logging
from config import CHECKPOINT_DIR
import tifffile as tiff
from PIL import Image
from unet_model import build_unet  # Ensure this imports the U-Net model correctly
from utils import load_rgb_images, save_mask_overlay
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(img_dir: str, mask_dir: str, img_height: int = 256, img_width: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses images and their corresponding masks.

    Args:
        img_dir (str): Directory containing HSI images.
        mask_dir (str): Directory containing mask images.
        img_height (int): Desired image height after resizing.
        img_width (int): Desired image width after resizing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of NumPy arrays containing images and masks.
    """
    try:
        # Supported image and mask extensions
        img_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')
        mask_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')

        # List all image and mask files
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(mask_extensions)]

        logging.info(f"Found {len(img_files)} HSI images and {len(mask_files)} mask images.")

        # Create a mapping from image base name to mask file
        mask_dict = {}
        for mask_file in mask_files:
            mask_base = os.path.splitext(mask_file)[0].lower()
            mask_dict[mask_base] = mask_file

        X = []
        Y = []
        matched = 0
        skipped_no_mask = 0
        skipped_load_error = 0

        for img_file in img_files:
            img_base = os.path.splitext(img_file)[0].lower()  # e.g., '011_hsi'
            
            # Extract the numerical part from the image filename
            # Assuming image filenames are in the format '###_hsi.tiff'
            if '_hsi' in img_base:
                num_part = img_base.split('_hsi')[0]  # e.g., '011'
            else:
                logging.warning(f"Image filename '{img_file}' does not contain '_hsi'. Skipping.")
                skipped_no_mask += 1
                continue

            # Construct the expected mask filename based on your naming convention
            # Masks are saved as '011-1.tiff', '012-1.tiff', etc.
            expected_mask_base = f"{num_part}-1"
            mask_file = mask_dict.get(expected_mask_base)

            if not mask_file:
                logging.warning(f"No mask found for image '{img_file}'. Skipping.")
                skipped_no_mask += 1
                continue

            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                # Load and preprocess the HSI image (31 channels)
                img = tiff.imread(img_path)  # Shape: (Bands, Height, Width)
                img = img.transpose(1, 2, 0)  # Convert to (Height, Width, Bands)
                img = tf.image.resize(img, [img_height, img_width]).numpy()
                img = img / np.max(img)  # Normalize based on max value
                X.append(img)

                # Load and preprocess the mask
                mask = Image.open(mask_path).convert("L")  # Grayscale
                mask = mask.resize((img_width, img_height))
                mask_array = np.array(mask) / 255.0  # Normalize to [0, 1]
                mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension
                Y.append(mask_array)

                matched += 1

            except Exception as e:
                logging.error(f"Error loading image/mask pair '{img_file}' and '{mask_file}': {e}")
                skipped_load_error += 1
                continue

        logging.info(f"Matched and loaded {matched} image-mask pairs.")
        if skipped_no_mask > 0:
            logging.info(f"Skipped {skipped_no_mask} images due to missing masks.")
        if skipped_load_error > 0:
            logging.info(f"Skipped {skipped_load_error} image-mask pairs due to loading errors.")

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
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
    image = load_rgb_images(image_path, target_size)
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
