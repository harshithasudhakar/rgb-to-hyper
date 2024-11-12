import os
import numpy as np
import cv2
import zipfile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
N_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 5
IMG_PATH = 'data\\unet\\images'  # Path to your images
MASK_PATH = 'data\\unet\\masks'    # Path to your masks
ZIP_PATH = 'data\\zips\\dataverse.zip'
MODEL_PATH = 'unet_microplastics.h5'


def extract_zip(zip_path, image_path, mask_path):
    # Ensure output directories exist
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for f in zip_ref.infolist():
            fname = f.filename
            # Check if the file is a PNG image
            if fname.endswith('.png'):
                if '-1.png' in fname:  # It's a mask
                    zip_ref.extract(f, mask_path)
                    print(f"Extracted {fname} to {mask_path}")
                else:  # It's an image
                    zip_ref.extract(f, image_path)
                    print(f"Extracted {fname} to {image_path}")

    # Load and preprocess the dataset


def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png'):  # Assuming images are .png files
            # Load image and resize
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)

            # Construct the mask filename based on image filename
            mask_filename = filename.replace('.png', '-1.png')
            mask_path = os.path.join(mask_dir, mask_filename)

            # Load mask and resize
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

            # Convert mask to one-hot encoding
            mask_one_hot = np.zeros((*mask.shape, N_CLASSES))
            for i in range(N_CLASSES):
                mask_one_hot[:, :, i] = (mask == i).astype(float)

            masks.append(mask_one_hot)

    return np.array(images), np.array(masks)

# Create U-Net model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), n_classes=N_CLASSES):
    inputs = layers.Input(input_size)

    # Encoder with reduced filters
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck with fewer filters
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)

    # Decoder with reduced filters
    up5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge5)

    up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge6)

    up7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge7)

    # Final layer with the number of classes
    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(conv7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data Augmentation
def augment_data(images, masks):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True,
                                 fill_mode='nearest')

    return datagen.flow(images, masks, batch_size=BATCH_SIZE)


# Main function to train the model
if __name__ == "__main__":
    extract_zip(zip_path=ZIP_PATH, image_path=IMG_PATH, mask_path=MASK_PATH)
    # Load data
    images, masks = load_data(IMG_PATH, MASK_PATH)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    # Create U-Net model
    model = unet_model()

    # Train the model with data augmentation
    train_generator = augment_data(X_train, y_train)
    model.fit(train_generator, epochs=EPOCHS,
              validation_data=(X_val, y_val), verbose=1)

    # Save the model
    model.save(MODEL_PATH)

    # Evaluate the model
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=-1)  # Convert to class labels
    y_val_classes = np.argmax(y_val, axis=-1)

    # Metrics
    print("Classification Report:")
    print(classification_report(y_val_classes.flatten(), y_pred_classes.flatten(
    ), target_names=['Background', 'Bead', 'Fiber', 'Fragment']))

    # Confusion Matrix
    conf_matrix = confusion_matrix(
        y_val_classes.flatten(), y_pred_classes.flatten())
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualize predictions
    for i in range(5):  # Display 5 examples
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(X_val[i])

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(y_val_classes[i], cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(y_pred_classes[i], cmap='gray')

        plt.show()
