import os
from PIL import Image
import numpy as np
import config
# Function to load RGB image


def load_rgb_image(file_path):
    return Image.open(file_path)

# Function to load HSI bands for a specific RGB image in .tiff format


def load_hsi_bands(folder_path, base_name):
    band_images = []
    for band_num in range(1, 32):  # Loop from 1 to 31 for the band numbers
        band_file = f"{base_name}_band_{band_num}.tiff"
        band_path = os.path.join(folder_path, band_file)

        if os.path.exists(band_path):
            band_images.append(np.array(Image.open(band_path)))
        else:
            print(f"Band file {band_file} not found in {folder_path}")
            return None  # Stop if any band file is missing

    return np.stack(band_images, axis=-1)  # Stack bands along the last axis


if __name__ == "__main__":
    rgb_to_hsi = {}

# Loop through each RGB image and find its corresponding HSI folder
    for rgb_file in os.listdir(config.RGB_IMAGE_PATH):
        if rgb_file.endswith('_clean.png'):
            rgb_path = os.path.join(config.RGB_IMAGE_PATH, rgb_file)

            # Extract base name (e.g., "ARAD_HS_0001" from "ARAD_HS_0001_clean.png")
            base_name = rgb_file.replace('_clean.png', '')
            hsi_folder = os.path.join(config.HSI_IMAGE_PATH, base_name)

            # Check if the corresponding HSI folder exists
            if os.path.isdir(hsi_folder):
                # Load RGB and HSI data
                rgb_image = load_rgb_image(rgb_path)
                hsi_bands = load_hsi_bands(hsi_folder, base_name)

                if hsi_bands is not None:  # Proceed only if all bands were loaded
                    # Map RGB to HSI
                    rgb_to_hsi[base_name] = {
                        'rgb': rgb_image, 'hsi': hsi_bands}
            else:
                print(f"No corresponding HSI folder found for {rgb_file}")

    # Example: Accessing an RGB and HSI pair
    example_key = list(rgb_to_hsi.keys())[0]
    rgb_image = rgb_to_hsi[example_key]['rgb']
    hsi_bands = rgb_to_hsi[example_key]['hsi']

    print(f"RGB Image Shape: {np.array(rgb_image).shape}")
    # Expected shape: (height, width, 31)
    print(f"HSI Bands Shape: {hsi_bands.shape}")
