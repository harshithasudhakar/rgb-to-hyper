import zipfile
import scipy.io
import numpy as np
import os
from PIL import Image
import gdown

# Step 1: Download the ZIP file from Google Drive
file_id = '1sPiNlWUiaDwmhkLyGQp3zGMRfmmjj1R9'  # Your file ID
output_zip_file = 'shared_file.zip'  # Name for the downloaded ZIP file
gdown.download(
    f'https://drive.google.com/uc?id={file_id}', output_zip_file, quiet=False)

# Step 2: Unzip and Process the .mat files
# Directory to save the extracted images
image_save_dir = '/kaggle/working/hsi_images'

# Create directory to save images
os.makedirs(image_save_dir, exist_ok=True)

# Open the ZIP file
with zipfile.ZipFile(output_zip_file, 'r') as zip_ref:
    mat_files = [f for f in zip_ref.namelist() if f.endswith('.mat')]

    # Iterate through each .mat file inside the ZIP
    for mat_file_name in mat_files:
        print(f"Processing {mat_file_name}")

        # Read .mat file in memory
        with zip_ref.open(mat_file_name) as mat_file:
            mat_data = scipy.io.loadmat(mat_file)

            # Extract the HSI data (modify the key as per your .mat structure)
            if 'HSI_data_key' in mat_data:  # Replace 'HSI_data_key' with the actual key
                hsi = mat_data['HSI_data_key']
            else:
                print(f"Key not found in {mat_file_name}")
                continue

            # Assuming hsi is a 3D array (height x width x bands)
            for i in range(hsi.shape[2]):  # Iterate over each band
                band_image = hsi[:, :, i]

                # Normalize the band image to [0, 255]
                band_image = (band_image - np.min(band_image)) / \
                    (np.max(band_image) - np.min(band_image)) * 255
                band_image = band_image.astype(np.uint8)

                # Save the band image as PNG
                image_name = os.path.basename(
                    mat_file_name).replace('.mat', f'band{i}.png')
                save_path = os.path.join(image_save_dir, image_name)
                Image.fromarray(band_image).save(save_path)
                print(f"Saved image: {save_path}")
