import os
import shutil
import zipfile
import logging
import numpy as np
import tifffile as tiff
from config import CHECKPOINT_DIR
from main import load_model_and_predict
from utils import (
    visualize_all_hsi_bands, 
    #create_hsi_grid_image,
    visualize_stacked_hsi,
    #visualize_false_color_composite,
    visualize_pca_composite,
    #visualize_pca_3d
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unet_main.log"),
        logging.StreamHandler()
    ]
)

# Define paths using raw strings
extract_dir = r"C:\Harshi\ECS-II\Dataset\extracted"
rgb_dir = r"C:\Harshi\ECS-II\Dataset\temp-rgb-micro"
mask_dir = r"C:\Harshi\ECS-II\Dataset\mask_micro"
zip_file_path = r"C:\Harshi\ECS-II\Dataset\dataverse_files full"

"""# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    logging.info(f"Extracted {zip_file_path} to {extract_dir}")

# Create the directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
logging.info(f"Ensured directories {rgb_dir} and {mask_dir} exist.")

# Iterate through the files in the extracted folder and move them into the correct directories
for root, _, files in os.walk(extract_dir):
    for filename in files:
        if filename.endswith(".png"):
            # Check if it's a mask image (ends with '-1.png')
            if filename.endswith("-1.png"):
                shutil.move(os.path.join(root, filename), os.path.join(mask_dir, filename))
                logging.info(f"Moved mask image {filename} to {mask_dir}")
            else:
                shutil.move(os.path.join(root, filename), os.path.join(rgb_dir, filename))
                logging.info(f"Moved RGB image {filename} to {rgb_dir}")

logging.info("Images sorted into 'rgb_micro' and 'mask_micro'.")

# List the contents of the directories
logging.info("Contents of 'rgb_micro':")
logging.info(os.listdir(rgb_dir))

logging.info("Contents of 'mask_micro':")
logging.info(os.listdir(mask_dir))"""

# Call load_model_and_predict with the sorted images
try:
    load_model_and_predict(
        rgb_path=rgb_dir,
        checkpoint_path=CHECKPOINT_DIR
    )
except Exception as e:
    logging.error(f"An error occurred during prediction: {str(e)}")

# Visualization of all HSI bands and stacking
try:
    # Visualize all bands
    hsi_data = visualize_all_hsi_bands(
        filepath=r"C:\Harshi\ECS-II\Dataset\gen_hsi\186_hsi.tiff",  # Use the specific TIFF file
        bands=None,              # Set to None to visualize all bands
        figsize=(25, 20)         # Adjust figsize as needed
    )
    
    if hsi_data.size != 0:
        # Stack bands into a single 3D NumPy array
        stacked_hsi = hsi_data  # hsi_data is already a 3D array (height, width, bands)
        logging.info(f"HSI data stacked with shape: {stacked_hsi.shape}")
        
        # Save the stacked HSI data as a multi-band TIFF
        output_stacked_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\stacked_hsi.tiff"
        try:
            tiff.imwrite(output_stacked_path, stacked_hsi)
            logging.info(f"Stacked HSI saved to: {output_stacked_path}")
        except Exception as e:
            logging.error(f"Failed to save stacked HSI: {e}")
        
        """
        # Create and save a grid image of all bands
        grid_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\HSI_Bands_Grid.png"
        try:
            create_hsi_grid_image(
                stacked_hsi=stacked_hsi,
                cols=5,
                figsize=(25, 20),
                save_path=grid_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during grid image creation: {e}")
        
        # False-Color Composite Visualization
        try:
            false_color_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\False_Color_Composite.png"
            visualize_false_color_composite(
                stacked_hsi=stacked_hsi,
                bands=[29, 19, 9],  # Example band indices for RGB channels
                figsize=(10, 10),
                save_path=false_color_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during False-Color Composite visualization: {e}")
        """
        # PCA Composite Visualization
        try:
            pca_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\PCA_Composite.png"
            visualize_pca_composite(
                stacked_hsi=stacked_hsi,
                n_components=3,
                figsize=(10, 10),
                save_path=pca_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during PCA Composite visualization: {e}")
        """
        # Interactive 3D PCA Visualization (Optional)
        try:
            visualize_pca_3d(
                stacked_hsi=stacked_hsi,
                n_components=3
            )
        except Exception as e:
            logging.error(f"An error occurred during 3D PCA visualization: {e}")
            """
        
        # Stacked IMG visualization
        try:
            visualize_stacked_hsi(stacked_hsi, save_path=r"C:\Harshi\ECS-II\Dataset\gen_hsi\stacked_hsi.tiff")
        except Exception as e:
            logging.error(f"An error occurred during stacked HSI visualization: {str(e)}")
    
    else:
        logging.error("HSI data is empty. Stacking and visualization skipped.")

except Exception as e:
    logging.error(f"An error occurred during HSI bands visualization and stacking: {str(e)}")
