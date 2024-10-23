import os
import scipy.io
import numpy as np
import tifffile as tiff
from scipy.io import loadmat


def extract_bands(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

    for mat_file in mat_files:
        # Full path of the .mat file
        mat_file_path = os.path.join(input_dir, mat_file)

        # Load the .mat file
        mat_data = loadmat(mat_file_path)

        # Extract the 'cube' data
        if 'cube' in mat_data:
            cube_data = mat_data['cube']

            # Create a folder for the current .mat file's bands
            mat_file_name = os.path.splitext(mat_file)[0]
            mat_output_folder = os.path.join(output_dir, mat_file_name)
            os.makedirs(mat_output_folder, exist_ok=True)

            # Loop through each band and save as a TIFF file
            for i in range(cube_data.shape[2]):
                band_data = cube_data[:, :, i]
                tiff_file_path = os.path.join(
                    mat_output_folder, f'{mat_file_name}_band_{i + 1}.tiff')
                tiff.imwrite(tiff_file_path, band_data.astype(
                    np.float32))  # Save the TIFF file
            print(
                f"Extracted {cube_data.shape[2]} bands from {mat_file} and saved as TIFF in {mat_output_folder}.")
        else:
            print(f"'cube' key not found in {mat_file}.")
