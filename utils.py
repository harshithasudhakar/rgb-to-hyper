# utils.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config import HSI_CHANNELS, IMG_WIDTH, IMG_HEIGHT

# Data Loading Functions


def load_rgb_images(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Load and resize RGB images from a specified directory."""
    images = []
    for filename in os.listdir(image_path):
        # Update to match the RGB naming convention
        if filename.endswith('_clean.png'):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(image_path, filename), target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(
                img) / 255.0  # Normalize
            images.append(img)
    return np.array(images)


def load_hsi_images_from_all_folders(base_folder, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Load and stack hyperspectral images from multiple folders."""
    all_hsi_images = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            images = []
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.tiff') or file.endswith('.tif'):
                    image_path = os.path.join(folder_path, file)
                    img = Image.open(image_path).convert(
                        'L')  # Convert to grayscale
                    img = img.resize(target_size)  # Resize to target size
                    # Add channel dimension
                    img = np.array(img)[..., np.newaxis]
                    images.append(img)
            if len(images) == 31:
                stacked_images = np.array(images)
                if stacked_images.shape == (31, IMG_HEIGHT, IMG_WIDTH, 1):
                    all_hsi_images.append(stacked_images.reshape(
                        IMG_HEIGHT, IMG_WIDTH, HSI_CHANNELS))
    return np.array(all_hsi_images)

# Loss Functions


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

# Evaluation Metrics


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
