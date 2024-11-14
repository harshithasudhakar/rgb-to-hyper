import os
import numpy as np
import tensorflow as tf
import zipfile
import logging
import random
import tifffile as tiff
from model import Generator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
from config import CHECKPOINT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
N_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 5
IMG_PATH = 'data\\unet\\images'
MASK_PATH = 'data\\unet\\masks'
HSI_PATH = 'data\\unet\\hsi'
ZIP_PATH = 'data\\zips\\dataverse.zip'
MODEL_PATH = 'unet_microplastics.h5'

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def load_rgb_images(image_path, img_height=256, img_width=256):
    """
    Load RGB images from the specified path.
    
    Args:
        image_path (str): Path to the directory containing the RGB images.
        img_height (int): Target height for resizing images.
        img_width (int): Target width for resizing images.
    
    Returns:
        np.ndarray: Array of loaded RGB images.
        list: List of image filenames without extensions.
    """
    if not os.path.isdir(image_path):
        raise ValueError(f"The provided path {image_path} is not a valid directory.")
    
    print(f"Loading RGB images from path: {image_path}")
    
    image_generator = ImageDataGenerator(rescale=1./255)
    image_files = [
        os.path.join(image_path, f) for f in os.listdir(image_path) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    rgb_images = []
    filenames = []
    
    for image_file in image_files:
        try:
            img = load_img(image_file, target_size=(img_height, img_width))
            img_array = img_to_array(img) * image_generator.rescale
            rgb_images.append(img_array)
            filenames.append(os.path.splitext(os.path.basename(image_file))[0])
            print(f"Loaded image: {image_file}")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    
    print(f"Total images loaded: {len(filenames)}")
    return np.array(rgb_images), filenames


def extract_zip(zip_path, image_path, mask_path):
    print("Extracting zip file:", zip_path)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for f in zip_ref.infolist():
            fname = f.filename
            if fname.endswith('.png'):
                if '-1.png' in fname:
                    zip_ref.extract(f, mask_path)
                    print(f"Extracted mask: {fname}")
                else:
                    zip_ref.extract(f, image_path)
                    print(f"Extracted image: {fname}")

def augment_data(images, masks):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True,
                                 fill_mode='nearest')

    return datagen.flow(images, masks, batch_size=BATCH_SIZE)

def visualize_results(model, X_val, y_val, num_examples=5):
    for i in range(num_examples):
        test_image = X_val[i]
        true_mask = y_val[i, ..., 0]  # Single-channel true mask

        # Predict the mask for the test image
        pred_mask = model.predict(np.expand_dims(test_image, axis=0))[0, ..., 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold to binary mask

        # Overlay the true and predicted masks on the image
        overlay_true = test_image.copy()
        overlay_pred = test_image.copy()

        # Apply red overlay for the mask area
        overlay_true[true_mask == 1] = [255, 0, 0]  # True mask overlay in red
        overlay_pred[pred_mask == 1] = [255, 0, 0]  # Predicted mask overlay in red

        # Plot the results
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(test_image)
        ax[0].set_title("Original Image")
        ax[1].imshow(overlay_true)
        ax[1].set_title("True Mask Overlay")
        ax[2].imshow(overlay_pred)
        ax[2].set_title("Predicted Mask Overlay")
        plt.show()

def unet_model(input_shape=(256, 256, 31)):
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
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_model_and_predict(rgb_path, checkpoint_path, output_path):
    print("Loading model and generating HSI images...")
    rgb_images, filenames = load_rgb_images(rgb_path)
    os.makedirs(output_path, exist_ok=True)
    
    if not filenames:
        logging.error("No valid RGB images to process.")
        return
    
    rgb_tensor = tf.convert_to_tensor(rgb_images, dtype=tf.float32)
    print("RGB images converted to tensor")
    
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join('.', checkpoint_path))
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        logging.info(f"Model restored from checkpoint: {latest_ckpt}")
        print("Model loaded successfully from checkpoint:", latest_ckpt)
    else:
        logging.error("No checkpoint found. Please check the checkpoint path.")
        return

    print("Generating HSI images...")
    generated_hsi = generator(rgb_tensor, training=False)
    

    expected_channels = 31
    actual_channels = generated_hsi.shape[-1]
    if actual_channels != expected_channels:
        logging.warning(f"Generated HSI has {actual_channels} channels instead of {expected_channels}.")
    
    for j in range(generated_hsi.shape[0]):
        try:
            # Clip and convert generated HSI image to float32
            hsi_image = tf.clip_by_value(generated_hsi[j], 0, 1).numpy().astype(np.float32)
            
            # Check if the last dimension has the expected channels
            if hsi_image.shape[-1] != expected_channels:
                logging.warning(f"Image {filenames[j]} has unexpected shape {hsi_image.shape}. Skipping.")
                continue
            
            # Define the output file path
            base_filename = os.path.splitext(filenames[j])[0]
            output_file = os.path.join(output_path, f"{base_filename}_hsi.tiff")
            
            # Save the HSI image as a TIFF file
            tiff.imwrite(output_file, hsi_image)
            logging.info(f"Saved HSI image: {output_file}")
            print(f"HSI image saved: {output_file}")
        
        except Exception as e:
            logging.error(f"Failed to save HSI image for {filenames[j]}: {str(e)}")
            print(f"Error saving HSI image for {filenames[j]}: {e}")

    # After all images are saved, randomly pick a few to visualize
    num_images_to_show = 5  # Choose how many images to visualize
    random_indices = random.sample(range(len(filenames)), num_images_to_show)

    # Visualize the randomly selected images using tifffile
    for idx in random_indices:
        base_filename = filenames[idx]
        output_file = os.path.join(output_path, f"{base_filename}_hsi.tiff")
        
        try:
            # Load the saved HSI TIFF file
            hsi_image = tiff.imread(output_file)
            
            # Optionally, display each channel or create a montage
            # Example: Display the first channel of the image
            tiff.imshow(hsi_image[:, :, 0], title=f"{base_filename} - Channel 1")
            
            # Example: Display a montage of all 31 channels
            grid_size = int(np.ceil(np.sqrt(expected_channels)))  # Closest square grid
            montage = np.zeros((grid_size * hsi_image.shape[0], grid_size * hsi_image.shape[1]))

            for i in range(expected_channels):
                row = i // grid_size
                col = i % grid_size
                montage[
                    row * hsi_image.shape[0]:(row + 1) * hsi_image.shape[0],
                    col * hsi_image.shape[1]:(col + 1) * hsi_image.shape[1]
                ] = hsi_image[:, :, i]

            # Display the montage
            tiff.imshow(montage, title=f"{base_filename} - All Channels Montage")
        
        except Exception as e:
            logging.error(f"Failed to read and visualize HSI image for {base_filename}: {str(e)}")
            print(f"Error visualizing HSI image for {base_filename}: {e}")

if __name__ == "__main__":
    print("Starting main process...")
    # extract_zip(ZIP_PATH, IMG_PATH, MASK_PATH)
    # rgb_img, rgb_fname = load_rgb_images(IMG_PATH)
    load_model_and_predict(rgb_path=IMG_PATH, checkpoint_path=CHECKPOINT_DIR, output_path=HSI_PATH)
    