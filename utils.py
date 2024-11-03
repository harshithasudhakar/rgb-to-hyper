import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from typing import List
import tensorflow as tf
from scipy.io import loadmat
from PIL import Image
from model import Generator, Discriminator
from config import HSI_CHANNELS, IMG_WIDTH, IMG_HEIGHT, OUT_DIR_PATH


def load_hsi_images(hsi_dir: str) -> dict:
    hsi_images = {}

    for folder in os.listdir(hsi_dir):
        folder_path = os.path.join(hsi_dir, folder)
        if os.path.isdir(folder_path):

            # Load each TIFF file in the current folder
            for tiff_file in os.listdir(folder_path):
                if tiff_file.endswith('.tiff') or tiff_file.endswith('.tif'):
                    image_path = os.path.join(folder_path, tiff_file)
                    hsi_images[tiff_file] = tiff.imread(image_path)
    print(f"There are a total of {len(hsi_images)} HSI images")
    return hsi_images


def load_tiff_images(tiff_dir):
    tiff_images = {}
    for folder in os.listdir(tiff_dir):
        folder_path = os.path.join(tiff_dir, folder)
        if os.path.isdir(folder_path):
            tiff_files = [f for f in os.listdir(
                folder_path) if f.endswith('.tiff')]
            for tiff_file in tiff_files:
                img = Image.open(os.path.join(folder_path, tiff_file))
                tiff_images[tiff_file] = np.array(img)

    return tiff_images


def map_hsi_to_rgb(hsi_images, band_wavelengths, rgb_images):
    mapped_rgb_images = {}
    rgb_wavelengths = {
        "red": 650,
        "green": 550,
        "blue": 450
    }

    for mat_file, hsi_cube in hsi_images.items():
        wavelengths = band_wavelengths[mat_file]
        red_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["red"]))
        green_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["green"]))
        blue_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["blue"]))

        # Print the indices and corresponding wavelengths for debugging
        print(f"Processing {mat_file}:")
        print(
            f"Red band index: {red_band_index}, Wavelength: {wavelengths[red_band_index]}")
        print(
            f"Green band index: {green_band_index}, Wavelength: {wavelengths[green_band_index]}")
        print(
            f"Blue band index: {blue_band_index}, Wavelength: {wavelengths[blue_band_index]}")

        # Extract RGB bands from the hyperspectral cube
        rgb_from_hsi = hsi_cube[:, :, [
            red_band_index, green_band_index, blue_band_index]]
        print(
            f"RGB from HSI min: {rgb_from_hsi.min()}, max: {rgb_from_hsi.max()}")
        rgb_scaled = np.clip(rgb_from_hsi * 255, 0, 255).astype(np.uint8)

        # Convert to RGB image
        hsi_rgb_image = Image.fromarray(rgb_scaled)
        base_name = mat_file.split('.')[0]
        rgb_file_name = f"{base_name}_clean.png"

        if rgb_file_name in rgb_images:
            rgb_image = Image.fromarray(rgb_images[rgb_file_name])
            # Resize if dimensions do not match
            if hsi_rgb_image.size != rgb_image.size:
                hsi_rgb_image = hsi_rgb_image.resize(
                    rgb_image.size, Image.ANTIALIAS)

            mapped_rgb_images[rgb_file_name] = np.array(hsi_rgb_image)

    return mapped_rgb_images


def check_normalization(hsi_images: dict):
    for fname, img in hsi_images.items():
        min_val = img.min()
        max_val = img.max()
        print(f"{fname}: Min = {min_val}, Max = {max_val}")


def load_rgb_images(rgb_path: str, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load RGB images in sorted order.

    Args:
        rgb_path: Path to RGB images directory
        target_size: (width, height) for resizing

    Returns:
        Tuple of (image array, filenames)
    """
    # Get sorted filenames to ensure consistent order
    filenames = sorted([f for f in os.listdir(rgb_path)
                       if f.endswith('_clean.png')])

    if not filenames:
        raise ValueError(f"No RGB images found in {rgb_path}")

    images = []
    valid_filenames = []

    for filename in filenames:
        try:
            img_path = os.path.join(rgb_path, filename)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)
            valid_filenames.append(filename)
        except Exception as e:
            logging.warning(f"Failed to load RGB image {filename}: {str(e)}")
            continue

    return np.array(images), valid_filenames


def load_paired_images(rgb_path: str, hsi_path: str):
    """
    Load both RGB and HSI images ensuring perfect correspondence.

    Args:
        rgb_path: Path to RGB images directory
        hsi_path: Path to HSI base directory
        target_size: (width, height) for resizing

    Returns:
        Tuple of (rgb_images, hsi_images) with guaranteed matching order
    """
    # First load RGB images and get filenames
    rgb_images, rgb_filenames = load_rgb_images(rgb_path)

    # Then load HSI images using RGB filenames to maintain order
    hsi_images = load_hsi_images_from_all_folders(hsi_path, rgb_filenames)

    # Verify we have matching counts
    if len(rgb_images) != len(hsi_images):
        raise ValueError(
            f"Mismatch in number of images: {len(rgb_images)} RGB vs {len(hsi_images)} HSI")

    return rgb_images, hsi_images


def load_hsi_images_from_all_folders(base_folder: str, rgb_filenames: List[str],
                                     target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load HSI images maintaining exact correspondence with RGB images.

    Args:
        base_folder: Path to HSI base directory containing subdirectories
        rgb_filenames: List of RGB filenames to match against
        target_size: (width, height) for resizing

    Returns:
        numpy array of HSI images in same order as RGB files
    """
    all_hsi_images = []
    expected_bands = 31

    # Process each RGB filename to find corresponding HSI folder
    for rgb_filename in rgb_filenames:
        # Extract the base name (e.g., 'ARAD_HS_0001' from 'ARAD_HS_0001_clean.png')
        folder_name = rgb_filename.replace('_clean.png', '')
        folder_path = os.path.join(base_folder, folder_name)

        if not os.path.isdir(folder_path):
            raise ValueError(
                f"HSI folder not found for RGB image {rgb_filename}")

        # Load all TIFF files from the folder
        tiff_files = sorted([f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.tiff', '.tif'))])

        if len(tiff_files) != expected_bands:
            raise ValueError(
                f"Expected {expected_bands} bands but found {len(tiff_files)} "
                f"in folder {folder_name}")

        # Load and process each band
        bands = []
        for tiff_file in tiff_files:
            img_path = os.path.join(folder_path, tiff_file)
            try:
                img = Image.open(img_path)
                img = img.resize(target_size)
                img_array = np.array(img)
                bands.append(img_array)
            except Exception as e:
                raise ValueError(
                    f"Failed to load HSI band {tiff_file}: {str(e)}")

        # Stack bands and reshape to HxWxC
        stacked_image = np.stack(bands)
        stacked_image = stacked_image.transpose(1, 2, 0)  # Reshape to HxWxC
        all_hsi_images.append(stacked_image)

    return np.array(all_hsi_images)


def save_mapped_images(mapped_rgb_images, OUT_DIR_PATHectory):
    for rgb_file_name, rgb_image in mapped_rgb_images.items():
        output_path = os.path.join(OUT_DIR_PATHectory, rgb_file_name)
        # Convert to PIL image and save
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(output_path)
        print(f"Saved: {output_path}")


def aggregate_weights():
    ckpt_list: list = ['./checkpoints/abdul_ckpt-2', './checkpoints/akshath_ckpt-2',
                       './checkpoints/harshitha_ckpt-5', './checkpoints/keshihan_ckpt-2',
                       './checkpoints/nihaal_ckpt-2', './checkpoints/hrithik_ckpt-2']

    gen = Generator()
    disc = Discriminator()

    num_ckpt = len(ckpt_list)
    checkpoint = tf.train.Checkpoint(generator=gen, discriminator=disc)

    gen_weights = np.zeros_like(gen.get_weights())
    disc_weights = np.zeros_like(disc.get_weights())

    for path in ckpt_list:
        checkpoint.restore(path).expect_partial()
        gen_weights = np.add(gen_weights, gen.get_weights())
        disc_weights = np.add(disc_weights, disc.get_weights())

    gen_weights = np.divide(gen_weights, num_ckpt)
    disc_weights = np.divide(disc_weights, num_ckpt)

    gen.set_weights(gen_weights)
    disc.set_weights(disc_weights)
    checkpoint.save('checkpoints/global_1ckpt')


def visualize_generated_images(rgb_batch, generated_hsi, hsi_batch, epoch, batch):
    # Assuming generated_hsi and hsi_batch are tensors in [batch_size, height, width, channels]
    fig, axes = plt.subplots(3, len(rgb_batch), figsize=(15, 5))

    for i in range(len(rgb_batch)):
        # Convert each tensor to a numpy array
        rgb_img = rgb_batch[i].numpy()
        gen_img = generated_hsi[i].numpy()
        real_img = hsi_batch[i].numpy()

        # Display RGB input
        axes[0, i].imshow(rgb_img)
        axes[0, i].axis('off')
        axes[0, i].set_title('RGB Input')

        # Display Original HSI as a composite image
        # Average intensity across all channels
        real_img_composite = real_img.mean(axis=-1)
        axes[1, i].imshow(real_img_composite, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Original HSI Composite')

        # Display Generated HSI as a composite image
        # Average intensity across all channels
        gen_img_composite = gen_img.mean(axis=-1)
        axes[2, i].imshow(gen_img_composite, cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Generated HSI Composite')

    plt.suptitle(f'Epoch {epoch}, Batch {batch}')

    # Save the plot to a file
    if not os.path.exists(OUT_DIR_PATH):
        print(f"Creating directory: {OUT_DIR_PATH}")
        os.makedirs(OUT_DIR_PATH, exist_ok=True)
    else:
        print(f"Directory already exists: {OUT_DIR_PATH}")

    file_path = os.path.join(OUT_DIR_PATH, f'epoch_{epoch}_batch_{batch}.png')
    print(f"Saving plot to: {file_path}")
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved successfully.")


def apply_paired_augmentation(rgb_batch, hsi_batch):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Ensure inputs are numpy arrays with correct shape
    rgb_np = rgb_batch.numpy()
    hsi_np = hsi_batch.numpy()

    # Get batch size and shapes
    batch_size = rgb_np.shape[0]
    augmented_rgb_list = []
    augmented_hsi_list = []

    # Process each image pair in the batch separately
    for i in range(batch_size):
        # Get single images
        rgb_img = rgb_np[i]  # Should be (height, width, channels)
        hsi_img = hsi_np[i]  # Should be (height, width, channels)

        # Combine for synchronized augmentation
        combined = np.concatenate([rgb_img, hsi_img], axis=-1)

        # Get and apply random transformation
        transform_parameters = data_gen.get_random_transform(rgb_img.shape)
        augmented_combined = data_gen.apply_transform(
            combined, transform_parameters)

        # Split back
        rgb_channels = rgb_img.shape[-1]
        augmented_rgb = augmented_combined[..., :rgb_channels]
        augmented_hsi = augmented_combined[..., rgb_channels:]

        augmented_rgb_list.append(augmented_rgb)
        augmented_hsi_list.append(augmented_hsi)

    # Stack back into batches
    augmented_rgb_batch = tf.stack(augmented_rgb_list)
    augmented_hsi_batch = tf.stack(augmented_hsi_list)

    return augmented_rgb_batch, augmented_hsi_batch


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

def npy_to_png(directory_path, output_directory, mode='rgb'):
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            npy_path = os.path.join(directory_path, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_directory, png_filename)
            data = np.load(npy_path)
            if mode == 'rgb':
                rgb_data = data[:, :, [0, 15, 30]]
                rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min()) * 255
                image_data = rgb_data.astype(np.uint8)
            elif mode == 'grayscale':
                grayscale_data = np.mean(data, axis=2)
                grayscale_data = (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min()) * 255
                image_data = grayscale_data.astype(np.uint8)
            else:
                raise ValueError("Invalid mode. Choose 'rgb' or 'grayscale'.")
            image = Image.fromarray(image_data)
            image.save(png_path)
            print(f"Saved {png_path}")
