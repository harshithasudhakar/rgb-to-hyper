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
EPOCHS = 2
IMG_PATH = r"C:\Harshi\ECS-II\Dataset\temp-gen-hsi"  # Path to your HSI images
MASK_PATH = r"C:\Harshi\ECS-II\Dataset\temp-mask"  # Path to your masks
MODEL_PATH = r"C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper"

class HSIGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, batch_size, img_height, img_width, desired_channels, shuffle=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.desired_channels = desired_channels
        self.shuffle = shuffle
        self.image_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))
        ])
        self.indexes = np.arange(len(self.image_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.image_files[k] for k in batch_indexes]

        X = []
        Y = []

        for file_name in batch_files:
            img_path = os.path.join(self.img_dir, file_name)
            
            # Construct mask filename with '.png' extension
            base_name = os.path.splitext(file_name)[0].replace('_hsi', '-1')  # e.g., '191_hsi.tiff' -> '191-1'
            mask_file = base_name + '.png'                                   # '191-1.png'
            mask_path = os.path.join(self.mask_dir, mask_file)

            try:
                # Load HSI image
                hsi = tiff.imread(img_path)  # Original shape: (31, 256, 256)
                if hsi.ndim != 3 or hsi.shape[0] != self.desired_channels:
                    logging.warning(f"Image '{file_name}' has unexpected shape {hsi.shape}. Skipping.")
                    continue
                hsi = np.transpose(hsi, (1, 2, 0))  # Transpose to (256, 256, 31)
                assert hsi.shape == (self.img_height, self.img_width, self.desired_channels), \
                    f"HSI image has incorrect shape: {hsi.shape}"
                X.append(hsi)

                # Load mask
                mask = Image.open(mask_path).convert('L')  # Convert to grayscale
                # Resize mask to (256, 256) using nearest-neighbor to preserve binary values
                if mask.size != (self.img_width, self.img_height):
                    mask = mask.resize((self.img_width, self.img_height), resample=Image.NEAREST)
                    logging.info(f"Resized mask '{mask_file}' to {(self.img_width, self.img_height)}")
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=-1)            # Shape: (256, 256, 1)
                mask = (mask > 0).astype(np.float32)            # Binarize
                assert mask.shape == (self.img_height, self.img_width, 1), \
                    f"Mask has incorrect shape: {mask.shape}"
                Y.append(mask)

                logging.info(f"Image '{file_name}' loaded with shape {hsi.shape}")
                
            except FileNotFoundError:
                logging.error(f"Mask file not found for image '{file_name}': '{mask_path}'. Skipping.")
            except AssertionError as ae:
                logging.error(f"Assertion error for '{file_name}': {ae}. Skipping.")
            except Exception as e:
                logging.error(f"Error loading '{file_name}' or its mask: {e}. Skipping.")

        X = np.array(X)
        Y = np.array(Y)

        if Y.size == 0:
            logging.warning(f"No valid masks found in batch {index}. Skipping this batch.")
            # Fetch the next batch to avoid empty Y
            return self.__getitem__((index + 1) % self.__len__())

        # Ensure that X and Y have the same number of samples
        assert X.shape[0] == Y.shape[0], "Number of images and masks do not match in batch."

        logging.info(f"Batch X shape: {X.shape}, Batch Y shape: {Y.shape}")

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_unet(input_shape=(256, 256, 31), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
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
