import os
import config
import numpy as np
import tensorflow as tf
from extract import extract_bands
from utils import pair_img, load_rgb_images, load_hsi_images_from_all_folders, discriminator_loss, generator_loss, mean_squared_error, peak_signal_to_noise_ratio, spectral_angle_mapper, visualize_generated_images
from model import Generator, Discriminator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Wasserstein Loss with Gradient Penalty for improved GAN training
def discriminator_loss(disc_real_output, disc_fake_output):
    return tf.reduce_mean(disc_fake_output) - tf.reduce_mean(disc_real_output)

def gradient_penalty(discriminator, real_data, fake_data):
    alpha = tf.random.uniform([real_data.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        disc_interpolated = discriminator(interpolated)
    gradients = gp_tape.gradient(disc_interpolated, [interpolated])[0]
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return gradient_penalty

# Custom perceptual loss function using a custom CNN model
def create_custom_perceptual_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Example architecture
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    model = tf.keras.Model(inputs, x)
    return model

def perceptual_loss(generated, real):
    perceptual_model = create_custom_perceptual_model(input_shape=(256, 256, 31))
    generated_features = perceptual_model(generated)
    real_features = perceptual_model(real)
    return tf.reduce_mean(tf.square(generated_features - real_features))

# Custom brightness adjustment to guide generated images to the expected brightness
def brightness_loss(generated, real):
    return tf.reduce_mean(tf.abs(tf.reduce_mean(generated, axis=[1, 2]) - tf.reduce_mean(real, axis=[1, 2])))

# Mean Absolute Error Loss function
def mae_loss(generated, real):
    return tf.reduce_mean(tf.abs(generated - real))

# Updated GAN training loop
def train_gan(rgb_images, hsi_images, generator, discriminator, mode="local"):
    # Initialize the checkpoint for saving model states
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                      discriminator_optimizer=discriminator_optimizer,
                                      generator=generator,
                                      discriminator=discriminator)

    rgb_images = tf.convert_to_tensor(rgb_images, dtype=tf.float32)
    hsi_images = tf.convert_to_tensor(hsi_images, dtype=tf.float32)

    for epoch in range(config.EPOCHS):
        for i in range(0, len(rgb_images), config.BATCH_SIZE):
            rgb_batch = rgb_images[i:i + config.BATCH_SIZE]
            hsi_batch = hsi_images[i:i + config.BATCH_SIZE]

            # Augment RGB batch images
            augmented_rgb_batch = next(data_gen.flow(rgb_batch.numpy(), batch_size=config.BATCH_SIZE))
            augmented_rgb_batch = tf.convert_to_tensor(augmented_rgb_batch, dtype=tf.float32)

            # Generate HSI and resize to target shape
            generated_hsi = generator(augmented_rgb_batch)
            target_shape = tf.shape(hsi_batch)[1:3]
            generated_hsi_resized = tf.image.resize(generated_hsi, target_shape)

            combined_real = tf.concat([hsi_batch, rgb_batch], axis=-1)
            combined_fake = tf.concat([generated_hsi_resized, augmented_rgb_batch], axis=-1)

            # Discriminator training with gradient penalty
            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real)
                disc_fake = discriminator(combined_fake)
                disc_loss = discriminator_loss(disc_real, disc_fake) + 10.0 * gradient_penalty(discriminator, combined_real, combined_fake)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Generator training with updated losses
            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch)
                generated_hsi_resized = tf.image.resize(generated_hsi, target_shape)
                combined_fake = tf.concat([generated_hsi_resized, augmented_rgb_batch], axis=-1)
                gen_loss = generator_loss(discriminator(combined_fake))

                # SAM, perceptual, brightness, and MAE losses
                sam_loss = spectral_angle_mapper(hsi_batch, generated_hsi_resized)
                perceptual = perceptual_loss(generated_hsi_resized, hsi_batch)
                brightness_diff = brightness_loss(generated_hsi_resized, hsi_batch)
                mae = mae_loss(generated_hsi_resized, hsi_batch)  # Add MAE loss

                # Adding perceptual, SAM, brightness, and MAE losses to generator loss
                gen_loss += 0.1 * sam_loss + 0.1 * perceptual + 0.05 * brightness_diff + 0.1 * mae

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            # Calculate metrics
            mse = mean_squared_error(hsi_batch, generated_hsi_resized)
            psnr = peak_signal_to_noise_ratio(hsi_batch, generated_hsi_resized)
            sam = spectral_angle_mapper(hsi_batch, generated_hsi_resized)

            visualize_generated_images(augmented_rgb_batch, generated_hsi, hsi_batch, epoch, i // config.BATCH_SIZE)

            print(f'Epoch: {epoch}, Batch: {i // config.BATCH_SIZE}, Disc Loss: {disc_loss.numpy()}, Gen Loss: {gen_loss.numpy()}, MSE: {mse.numpy()}, PSNR: {psnr.numpy()}, SAM: {sam.numpy()}')
        
        # Save the model checkpoint
        checkpoint.save(file_prefix=checkpoint_path)

if __name__ == "__main__":
    mode = "global"
    data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                                  zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

    # Load images
    rgb_images = load_rgb_images(config.RGB_IMAGE_PATH)
    hsi_images = load_hsi_images_from_all_folders(config.HSI_IMAGE_PATH)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Define learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE, decay_steps=1000, decay_rate=0.96)

    # Optimizers with learning rate scheduler
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=config.BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=config.BETA_1)

    log_dir = config.LOG_DIR
    summary_writer = tf.summary.create_file_writer(log_dir)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.GLOBAL_CHECKPOINT_PREFIX) if mode == "global" else os.path.join(config.CHECKPOINT_DIR, config.LOCAL_CHECKPOINT_PREFIX)

    # Start training
    train_gan(rgb_images, hsi_images, generator=generator, discriminator=discriminator, mode=mode)
