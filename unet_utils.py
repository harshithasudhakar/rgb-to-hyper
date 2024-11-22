## Implementing Single Image Segmentation Prediction in `unet_main.py`

#To perform segmentation overlay on a single RGB image using your trained U-Net model, follow the steps below. This involves modifying `unet_main.py` to accept a single image input, perform prediction, and overlay the segmentation mask on the original image.

### **Step 1: Update `load_model_and_predict` Function in `main.py`**

#Ensure that the `load_model_and_predict` function can handle both single images and batches. Modify it to accept an optional parameter for single image paths.

import tensorflow as tf
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from config import CHECKPOINT_DIR
import tifffile as tiff
from PIL import Image
from unet_model import build_unet  # Ensure this imports the U-Net model correctly
from utils import load_rgb_images, save_mask_overlay
from typing import List, Tuple
import cv2
from sklearn.decomposition import PCA
import matplotlib.cm as cm

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
                # img = img.transpose(1, 2, 0)  # Convert to (Height, Width, Bands)
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
                logging.error(f"Error loading image or mask pair '{img_file}' and '{mask_file}': {e}")
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
    
def load_model(checkpoint_path: str) -> tf.keras.Model:
    """
    Load the trained U-Net model from the .keras checkpoint file.

    Args:
        checkpoint_path (str): Full path to the 'best_model.keras' file.

    Returns:
        tf.keras.Model: Loaded U-Net model.
    """
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        logging.info(f"Model loaded from: {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading model from {checkpoint_path}: {e}")
        raise FileNotFoundError("Checkpoint not found.")
    return model

# unet_utils.py

import tifffile as tiff

def preprocess_image(image_path: str, target_size=(256, 256)) -> np.ndarray:
    """
    Load and preprocess the HSI image.

    Args:
        image_path (str): Path to the HSI image.
        target_size (tuple): Desired image size.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        # Read the multi-band TIFF image
        img = tiff.imread(image_path)  # Shape: (Bands, Height, Width) or (Height, Width, Bands)
        
        # Check the shape and transpose if necessary
        if img.ndim == 3 and img.shape[0] == 31:
            # If shape is (Bands, Height, Width), transpose to (Height, Width, Bands)
            img = img.transpose(1, 2, 0)
            logging.info(f"Image transposed to shape: {img.shape}")
        elif img.ndim == 3 and img.shape[2] != 31:
            logging.warning(f"Expected 31 channels, but got {img.shape[2]} channels.")
        
        # Convert to float32 if not already
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            logging.info(f"Image dtype converted to float32.")
        
        # Resize the image to the target size
        img = tf.image.resize(img, target_size).numpy()
        logging.info(f"Image resized to: {img.shape[:2]}")
        
        # Normalize the image based on the max value in the image
        max_val = np.max(img)
        if max_val > 0:
            img /= max_val
            logging.info(f"Image normalized by max value: {max_val}")
        else:
            logging.warning("Max value of the image is 0. Skipping normalization.")
        
        # Add a batch dimension
        img = np.expand_dims(img, axis=0)  # Shape: (1, Height, Width, Bands)
        logging.info(f"Image shape after adding batch dimension: {img.shape}")
        
        return img
    
    except Exception as e:
        logging.error(f"Error preprocessing image '{image_path}': {e}")
        raise e
    

def predict_mask(model: tf.keras.Model, image: np.ndarray, output_dir: str) -> np.ndarray:
    """
    Predict the segmentation mask using the U-Net model.

    Args:
        model (tf.keras.Model): Loaded U-Net model.
        image (np.ndarray): Preprocessed image array.
        output_dir (str): Directory to save the mask images.

    Returns:
        np.ndarray: Predicted binary mask.
    """
    try:
        prediction = model.predict(image)
        logging.info(f"Prediction shape: {prediction.shape}")

        # Save raw prediction
        raw_pred = prediction[0, :, :, 0]  # Shape: (256, 256)
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_pred, cmap='viridis')
        plt.colorbar()
        raw_pred_path = os.path.join(output_dir, "raw_prediction.png")
        plt.title("Raw Model Prediction")
        plt.savefig(raw_pred_path)
        plt.close()
        logging.info(f"Raw prediction saved to {raw_pred_path}")

        # Apply sigmoid activation if not already applied
        if prediction.max() > 1.0:
            prediction = tf.keras.activations.sigmoid(prediction).numpy()
            logging.info("Applied sigmoid activation to predictions.")

        # Binarize the mask using a threshold of 0.5
        mask = (prediction > 0.5).astype(np.uint8)
        logging.info("Binarized the mask with threshold 0.5.")

        # Remove unnecessary dimensions
        mask = np.squeeze(mask)
        logging.info(f"Mask shape after squeezing: {mask.shape}")

        # Log mask statistics
        logging.info(f"Mask Statistics - Min: {mask.min()}, Max: {mask.max()}, Unique Values: {np.unique(mask)}")

        # Save the mask as an image for inspection
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode='L')
        mask_path = os.path.join(output_dir, "predicted_mask.png")
        mask_image.save(mask_path)
        logging.info(f"Saved predicted mask to {mask_path}")

        return mask
    except Exception as e:
        logging.error(f"Error during mask prediction: {e}")
        raise e


def overlay_mask(image_path: str, mask: np.ndarray, output_path: str):
    """
    Overlay the predicted mask on the original HSI image converted to grayscale.

    Args:
        image_path (str): Path to the original HSI image.
        mask (np.ndarray): Predicted binary mask.
        output_path (str): Path to save the overlaid image.

    Returns:
        None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert HSI to Grayscale for visualization
        img = Image.open(image_path)
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((mask.shape[1], mask.shape[0]))
        logging.info(f"Grayscale image resized to: {img.size}")
        
        # Convert mask to RGB with colormap for better visibility
        cmap = cm.get_cmap('jet')  # Choose a colormap
        mask_colored = cmap(mask)[:, :, :3]  # RGB channels
        mask_colored = (mask_colored * 255).astype(np.uint8)
        mask_rgb = Image.fromarray(mask_colored, mode='RGB')
        
        # Convert grayscale image to RGB for blending
        img_rgb = img.convert("RGB")
        
        # Blend the grayscale RGB image with the colored mask
        overlay = Image.blend(img_rgb, mask_rgb, alpha=0.5)
        
        overlay.save(output_path)
        logging.info(f"Overlay image saved to: {output_path}")
    
    except Exception as e:
        logging.error(f"Error overlaying mask: {e}")
        raise e

def visualize_segmentation(image_path: str, mask: np.ndarray, output_path: str):
    """
    Visualize the original HSI image, predicted mask, and their overlay.

    Args:
        image_path (str): Path to the original HSI TIFF file.
        mask (np.ndarray): Predicted binary mask with shape (Height, Width).
        output_path (str): Path to save the overlay visualization.

    Returns:
        None
    """
    try:
        # Load the HSI image using tifffile
        hsi = tiff.imread(image_path)  # Shape: (Bands, Height, Width) or (Height, Width, Bands)
        logging.info(f"Loaded HSI image shape: {hsi.shape}, dtype: {hsi.dtype}")

        # Transpose if necessary to get (Height, Width, Bands)
        if hsi.ndim == 3 and hsi.shape[0] == 31:
            hsi = hsi.transpose(1, 2, 0)  # Now shape is (Height, Width, Bands)
            logging.info(f"HSI image transposed to shape: {hsi.shape}")

        # Apply PCA to convert HSI to RGB
        rgb_image = hsi_to_rgb_pca(hsi)  # Shape: (Height, Width, 3)
        logging.info(f"PCA-based RGB image shape: {rgb_image.shape}")

        # Normalize RGB image for display
        rgb_image_normalized = rgb_image.copy()
        rgb_image_normalized -= rgb_image_normalized.min()
        rgb_image_normalized /= rgb_image_normalized.max()

        # Create a figure with three subplots
        plt.figure(figsize=(18, 6))

        # Original HSI RGB Image (PCA-based)
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image_normalized)
        plt.title('Original HSI Image (PCA RGB)')
        plt.axis('off')

        # Predicted Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        # Overlay Mask on HSI Image
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_image_normalized)
        
        # Create a colored mask for better visibility (e.g., red)
        colored_mask = np.zeros_like(rgb_image_normalized)
        colored_mask[..., 0] = mask  # Assign mask to the Red channel
        
        # Normalize mask for visualization
        mask_normalized = mask / mask.max()

        plt.imshow(colored_mask, cmap='Reds', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        logging.info(f"Segmentation visualization saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error in visualize_segmentation: {e}")
        raise e

def visualize_overlay(hsi_path: str, mask: np.ndarray, output_path: str):
    """
    Visualize and save the overlay of the predicted mask on the HSI image in grayscale with red highlights.

    Args:
        hsi_path (str): Path to the original HSI image.
        mask (np.ndarray): Predicted binary mask with shape (Height, Width).
        output_path (str): Path to save the overlaid image.

    Returns:
        None
    """
    try:
        # Load the HSI image
        hsi = tiff.imread(hsi_path)
        logging.info(f"Loaded HSI image from {hsi_path} with shape {hsi.shape} and dtype {hsi.dtype}")

        # Ensure HSI has shape (Height, Width, Bands)
        if hsi.ndim != 3 or hsi.shape[2] != 31:
            raise ValueError(f"HSI image has invalid shape {hsi.shape}. Expected shape (Height, Width, 31).")

        # Convert HSI to grayscale by averaging across spectral bands
        grayscale_image = np.mean(hsi, axis=2)
        logging.info(f"Converted HSI to grayscale with shape {grayscale_image.shape}")

        # Normalize the grayscale image to [0, 1] for visualization
        grayscale_min = grayscale_image.min()
        grayscale_max = grayscale_image.max()
        if grayscale_max - grayscale_min == 0:
            raise ValueError("Cannot normalize image with zero dynamic range.")
        grayscale_normalized = (grayscale_image - grayscale_min) / (grayscale_max - grayscale_min)
        logging.info("Normalized grayscale image to [0, 1] range.")

        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 0.5).astype(np.uint8)
            logging.info("Converted mask to binary.")

        # Resize mask if it doesn't match the grayscale image dimensions
        if mask.shape != grayscale_normalized.shape:
            mask = cv2.resize(mask, (grayscale_normalized.shape[1], grayscale_normalized.shape[0]), interpolation=cv2.INTER_NEAREST)
            logging.info(f"Resized mask to match grayscale image dimensions: {mask.shape}")

        # Log mask statistics
        logging.info(f"Mask Statistics - Min: {mask.min()}, Max: {mask.max()}, Unique Values: {np.unique(mask)}")

        # Verify dimensions
        assert mask.shape == grayscale_normalized.shape, "Mask and grayscale image dimensions do not match."

        # Normalize the grayscale image to [0, 1] for visualization
        grayscale_min = grayscale_image.min()
        grayscale_max = grayscale_image.max()
        if grayscale_max - grayscale_min == 0:
            raise ValueError("Cannot normalize image with zero dynamic range.")
        grayscale_normalized = (grayscale_image - grayscale_min) / (grayscale_max - grayscale_min)
        logging.info("Normalized grayscale image to [0, 1] range.")

        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 0.5).astype(np.uint8)
            logging.info("Converted mask to binary.")

        # Resize mask if it doesn't match the grayscale image dimensions
        if mask.shape != grayscale_normalized.shape:
            mask = cv2.resize(mask, (grayscale_normalized.shape[1], grayscale_normalized.shape[0]), interpolation=cv2.INTER_NEAREST)
            logging.info(f"Resized mask to match grayscale image dimensions: {mask.shape}")

        # Debug: Check mask statistics
        logging.info(f"Mask statistics - min: {mask.min()}, max: {mask.max()}, unique values: {np.unique(mask)}")

        # Create a red mask
        red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        red_mask[:, :, 0] = mask  # Red channel

        # Convert grayscale to RGB
        grayscale_rgb = np.stack([grayscale_normalized]*3, axis=2)

        # Define alpha for blending
        alpha = 0.5  # Adjust this value as needed (0.0 transparent, 1.0 opaque)

        # Blend the grayscale RGB image with the red mask
        overlay = grayscale_rgb.copy()
        overlay[mask == 1] = (1 - alpha) * grayscale_rgb[mask == 1] + alpha * red_mask[mask == 1]

        logging.info("Applied red mask overlay on grayscale RGB image.")

        # Convert overlay to uint8
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        # Save the overlay image using PIL
        overlay_image = Image.fromarray(overlay_uint8)
        overlay_image.save(output_path)
        logging.info(f"Overlay visualization saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in visualize_overlay: {e}", exc_info=True)
        raise e
    
    
def load_model_and_predict(image_path: str, checkpoint_path: str, output_path: str) -> np.ndarray:
    """
    Load the model, predict the mask for a single image, visualize, and save the overlay.

    Args:
        image_path (str): Path to the input HSI image.
        checkpoint_path (str): Path to the saved model.
        output_path (str): Path to save the overlay image.

    Returns:
        np.ndarray: Predicted binary mask.
    """
    try:
        # Load the model
        model = load_model(checkpoint_path)

        # Preprocess the image
        image = preprocess_image(image_path)
        logging.info(f"Image loaded and preprocessed: {image_path}")

        # Predict the mask
        mask = predict_mask(model, image, os.path.dirname(output_path))
        logging.info("Mask prediction completed.")

        # Visualize the overlay with red highlights
        visualize_overlay(image_path, mask, output_path)
        logging.info(f"Overlay saved to: {output_path}")

        return mask  # Return the mask for further use if needed

    except Exception as e:
        logging.error(f"Error in load_model_and_predict: {e}")
        return None
    
def hsi_to_rgb_pca(hsi_image: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Convert HSI image to RGB using PCA.

    Args:
        hsi_image (np.ndarray): HSI image array with shape (Height, Width, Bands).
        n_components (int): Number of principal components to retain.

    Returns:
        np.ndarray: RGB image array with shape (Height, Width, 3).
    """
    try:
        height, width, bands = hsi_image.shape
        hsi_reshaped = hsi_image.reshape(-1, bands)  # Shape: (Height*Width, Bands)
        
        pca = PCA(n_components=n_components)
        rgb_flat = pca.fit_transform(hsi_reshaped)
        
        # Normalize PCA output to [0, 1]
        rgb_flat -= rgb_flat.min()
        rgb_flat /= rgb_flat.max()
        
        rgb_image = rgb_flat.reshape(height, width, n_components)
        return rgb_image

    except Exception as e:
        logging.error(f"Error converting HSI to RGB using PCA: {e}")
        raise e
    
def display_mask(mask: np.ndarray):
    """
    Display the predicted mask.

    Args:
        mask (np.ndarray): Predicted binary mask.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    plt.show()

def create_synthetic_mask(height: int, width: int) -> np.ndarray:
    """
    Create a synthetic binary mask with a white rectangle.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        np.ndarray: Synthetic binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (200, 200), 1, -1)  # White rectangle
    return mask


def visualize_synthetic_overlay(image_path: str, output_path: str) -> None:
    """
    Create and overlay a synthetic mask on the grayscale image.

    Args:
        image_path (str): Path to the input HSI image.
        output_path (str): Path to save the overlaid image.

    Returns:
        None
    """
    try:
        # Load HSI image
        hsi = tiff.imread(image_path)
        logging.info(f"Loaded HSI image from {image_path} with shape {hsi.shape} and dtype {hsi.dtype}")

        # Ensure HSI has shape (Height, Width, Bands)
        if hsi.ndim != 3 or hsi.shape[2] != 31:
            raise ValueError(f"HSI image has invalid shape {hsi.shape}. Expected shape (Height, Width, 31).")

        # Convert HSI to grayscale by averaging across spectral bands
        grayscale_image = np.mean(hsi, axis=2)
        logging.info(f"Converted HSI to grayscale with shape {grayscale_image.shape}")

        # Normalize the grayscale image to [0, 1] for visualization
        grayscale_min = grayscale_image.min()
        grayscale_max = grayscale_image.max()
        if grayscale_max - grayscale_min == 0:
            raise ValueError("Cannot normalize image with zero dynamic range.")
        grayscale_normalized = (grayscale_image - grayscale_min) / (grayscale_max - grayscale_min)
        logging.info("Normalized grayscale image to [0, 1] range.")

        # Create synthetic mask
        synthetic_mask = create_synthetic_mask(grayscale_image.shape[0], grayscale_image.shape[1])
        logging.info("Synthetic mask created.")

        # Log mask statistics
        logging.info(f"Synthetic Mask Statistics - Min: {synthetic_mask.min()}, Max: {synthetic_mask.max()}, Unique Values: {np.unique(synthetic_mask)}")

        # Create a red mask
        red_channel = synthetic_mask.astype(np.float32)  # Red channel
        red_mask = np.zeros((synthetic_mask.shape[0], synthetic_mask.shape[1], 3), dtype=np.float32)
        red_mask[:, :, 0] = red_channel  # Red channel
        logging.info("Created red mask.")

        # Convert grayscale to RGB
        grayscale_rgb = np.stack([grayscale_normalized]*3, axis=2)
        logging.info("Converted grayscale image to RGB.")

        # Define alpha for blending
        alpha = 0.5  # Adjust this value as needed (0.0 transparent, 1.0 opaque)

        # Blend the grayscale RGB image with the red mask
        overlay = np.copy(grayscale_rgb)
        overlay[synthetic_mask == 1] = (1 - alpha) * grayscale_rgb[synthetic_mask == 1] + alpha * red_mask[synthetic_mask == 1]
        logging.info("Blended red mask with grayscale RGB image.")

        # Convert overlay to uint8
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        # Save the overlay image using PIL
        overlay_image = Image.fromarray(overlay_uint8)
        overlay_image.save(output_path)
        logging.info(f"Synthetic overlay visualization saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in visualize_synthetic_overlay: {e}")
