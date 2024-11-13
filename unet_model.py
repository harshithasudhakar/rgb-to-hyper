import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16
EPOCHS = 50
IMG_PATH = r'c:\Harshi\ECS-II\Dataset\rgb_micro'  # Path to your images
MASK_PATH = r'c:\Harshi\ECS-II\Dataset\mask_micro'  # Path to your masks
MODEL_PATH = 'unet_microplastics.h5'

# Load and preprocess the dataset
def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.npy'):
            image = np.load(os.path.join(image_dir, filename))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)

            mask_filename = filename.replace('.npy', '_mask.npy')
            mask = np.load(os.path.join(mask_dir, mask_filename))
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

            # Convert to binary mask (single channel, 0 = background, 1 = target)
            mask_binary = (mask > 0).astype(np.float32)  # Assuming non-zero pixels are the target
            masks.append(mask_binary[..., np.newaxis])  # Add a channel dimension

    return np.array(images), np.array(masks)

# Define U-Net model
def unet_model(input_shape=(256, 256, 3)):
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

# Data Augmentation
def augment_data(images, masks):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True,
                                 fill_mode='nearest')

    return datagen.flow(images, masks, batch_size=BATCH_SIZE)

# Visualize results with side-by-side comparison
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

# Main function to train the model
if __name__ == "__main__":
    # Load data
    images, masks = load_data(IMG_PATH, MASK_PATH)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Create U-Net model
    model = unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Train the model with data augmentation
    train_generator = augment_data(X_train, y_train)
    model.fit(train_generator, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1)

    # Save the model
    model.save(MODEL_PATH)

    # Visualize predictions
    visualize_results(model, X_val, y_val, num_examples=5)
