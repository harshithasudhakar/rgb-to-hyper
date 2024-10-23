import os
import numpy as np
import tifffile as tiff
import tensorflow as tf
from PIL import Image
from config import HSI_CHANNELS, IMG_WIDTH, IMG_HEIGHT


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


def load_rgb_images(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    images = []
    for filename in os.listdir(image_path):
        if filename.endswith('_clean.png'):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(image_path, filename), target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(
                img) / 255.0
            images.append(img)
    return np.array(images)


def load_hsi_images_from_all_folders(base_folder, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    all_hsi_images = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            images = []
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.tiff') or file.endswith('.tif'):
                    image_path = os.path.join(folder_path, file)
                    img = Image.open(image_path).convert(
                        'L')
                    img = img.resize(target_size)
                    img = np.array(img)[..., np.newaxis]
                    images.append(img)
            if len(images) == 31:
                stacked_images = np.array(images)
                if stacked_images.shape == (31, IMG_HEIGHT, IMG_WIDTH, 1):
                    all_hsi_images.append(stacked_images.reshape(
                        IMG_HEIGHT, IMG_WIDTH, HSI_CHANNELS))
    return np.array(all_hsi_images)


def save_mapped_images(mapped_rgb_images, output_directory):
    for rgb_file_name, rgb_image in mapped_rgb_images.items():
        output_path = os.path.join(output_directory, rgb_file_name)
        # Convert to PIL image and save
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(output_path)
        print(f"Saved: {output_path}")


def pair_img(hsi_dir: str, tiff_dir: str, rgb_dir: str, map_dir: str):
    hsi_img, band_wvl = load_hsi_images(hsi_dir)
    tiff_img = load_tiff_images(tiff_dir)
    rgb_img = load_rgb_images(rgb_dir)

    mapped_img = map_hsi_to_rgb(hsi_img, band_wvl, rgb_img)
    save_mapped_images(mapped_img, map_dir)
    print("Mapped RGB images (from HSI):", mapped_img)
    print("Loaded RGB images:", rgb_img)
    print("Loaded TIFF images:", tiff_img)


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_output), fake_output)
    return tf.reduce_mean(real_loss + fake_loss)


def generator_loss(fake_output):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))


def pixel_loss(generated, target):
    return tf.reduce_mean(tf.square(generated - target))


def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def spectral_angle_mapper(y_true, y_pred, epsilon=1e-8):
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.norm(y_true, axis=-1)
    norm_pred = tf.norm(y_pred, axis=-1)
    cos_theta = dot_product / (tf.maximum(norm_true * norm_pred, epsilon))
    cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
    return tf.reduce_mean(tf.acos(cos_theta))
