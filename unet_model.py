# unet_model.py

import logging
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tifffile as tiff  # Import tifffile for reading TIFF files
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate # type: ignore

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
N_CLASSES = 1  # Binary segmentation
BATCH_SIZE = 16
EPOCHS = 5
IMG_PATH = r"C:\Harshi\ECS-II\Dataset\temp-gen-hsi"  # Path to your HSI images
MASK_PATH = r"C:\Harshi\ECS-II\Dataset\temp-mask"  # Path to your masks
MODEL_PATH = r"C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper"

class HSIGenerator(Sequence):
    def __init__(
        self,
        img_dir,
        mask_dir,
        batch_size=16,
        img_height=256,
        img_width=256,
        desired_channels=31,
        shuffle=True
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.desired_channels = desired_channels
        self.shuffle = shuffle
        self.img_files = [
            f for f in os.listdir(img_dir) if f.lower().endswith(('.tiff', '.tif'))
        ]
        self.mask_files = [
            f for f in os.listdir(mask_dir) if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))
        ]
        self.mask_dict = {os.path.splitext(f)[0].lower(): f for f in self.mask_files}
        self.indexes = np.arange(len(self.img_files))
        self.on_epoch_end()

        logging.info(f"Initialized HSIGenerator with {len(self.img_files)} images.")

    def __len__(self):
        length = int(np.floor(len(self.img_files) / self.batch_size))
        logging.debug(f"Number of batches per epoch: {length}")
        return length

    def __getitem__(self, index):
        # Generate indexes for the batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]

        # Retrieve the filenames for the batch
        batch_imgs = [self.img_files[k] for k in batch_indexes]

        logging.info(f"Generating batch {index + 1}: {batch_imgs}")

        # Generate data
        X, Y = self.__data_generation(batch_imgs)

        logging.debug(f"Batch {index + 1} - X shape: {X.shape}, Y shape: {Y.shape}")

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            logging.info("Shuffled the data indices for the new epoch.")

    def __data_generation(self, batch_imgs):
        X = []
        Y = []
        for img_file in batch_imgs:
            img_base = os.path.splitext(img_file)[0].lower()
            if '_hsi' in img_base:
                num_part = img_base.split('_hsi')[0]
            else:
                logging.warning(f"Image filename '{img_file}' does not contain '_hsi'. Skipping.")
                continue  # Skip files without '_hsi'

            expected_mask_base = f"{num_part}-1"
            mask_file = self.mask_dict.get(expected_mask_base)
            if not mask_file:
                logging.warning(f"No mask found for image '{img_file}'. Skipping.")
                continue  # Skip if no corresponding mask

            img_path = os.path.join(self.img_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            try:
                # Load and preprocess HSI image
                img = tiff.imread(img_path)  # Shape: (Bands, Height, Width)
                bands = img.shape[2]
                logging.debug(f"Image '{img_file}' loaded with shape {img.shape}")

                if bands < self.desired_channels:
                    logging.warning(
                        f"Image '{img_file}' has {bands} channels, less than desired {self.desired_channels}. Skipping."
                    )
                    continue  # Skip if insufficient channels
                elif bands > self.desired_channels:
                    img = img[:self.desired_channels, :, :]  # Retain the first 'desired_channels' bands
                    logging.debug(f"Image '{img_file}' channels reduced to {self.desired_channels}.")


                logging.debug(f"Image '{img_file}' shape after transpose: {img.shape}")

                img = img.astype('float32')
                img_max = np.max(img)
                if img_max == 0:
                    logging.warning(f"Image '{img_file}' has maximum value 0 after loading. Skipping.")
                    continue
                img = img / img_max  # Normalize based on max value
                img = np.clip(img, 0, 1)  # Ensure values are within [0, 1]
                X.append(img)

                # Load and preprocess mask
                mask = Image.open(mask_path).convert("L")  # Grayscale
                mask = mask.resize((self.img_width, self.img_height))
                mask_array = np.array(mask).astype('float32') / 255.0  # Normalize to [0, 1]
                mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension
                Y.append(mask_array)

                logging.debug(f"Processed image and mask for '{img_file}'.")

            except Exception as e:
                logging.error(f"Error loading image/mask pair '{img_file}' and '{mask_file}': {e}")
                continue

        X = np.array(X)
        Y = np.array(Y)

        logging.debug(f"Generated data - X shape: {X.shape}, Y shape: {Y.shape}")

        return X, Y

def build_unet(input_shape=(256, 256, 31), num_classes=1):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output Layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
