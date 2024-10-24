# main.py
import os
import config
import numpy as np
import tensorflow as tf
from extract import extract_bands
from utils import pair_img, load_rgb_images, load_hsi_images_from_all_folders, discriminator_loss, generator_loss, mean_squared_error, peak_signal_to_noise_ratio, spectral_angle_mapper
from model import Generator, Discriminator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_gan(rgb_images, hsi_images, generator: Generator, discriminator: Discriminator, mode="local"):
    if mode == "global":
        checkpoint = tf.train.Checkpoint(
            generator=generator, discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint('./checkpoints/'))
    else:
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

            augmented_rgb_batch = next(data_gen.flow(
                rgb_batch.numpy(), batch_size=config.BATCH_SIZE))
            augmented_rgb_batch = tf.convert_to_tensor(
                augmented_rgb_batch, dtype=tf.float32)

            generated_hsi = generator(augmented_rgb_batch)

            target_shape = tf.shape(hsi_batch)[1:3]
            generated_hsi_resized = tf.image.resize(
                generated_hsi, target_shape)
            augmented_rgb_batch_resized = tf.image.resize(
                augmented_rgb_batch, target_shape)

            combined_real = tf.concat([hsi_batch, rgb_batch], axis=-1)
            combined_fake = tf.concat(
                [generated_hsi_resized, augmented_rgb_batch_resized], axis=-1)

            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real)
                disc_fake = discriminator(combined_fake)
                disc_loss = discriminator_loss(disc_real, disc_fake)

            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch)
                generated_hsi_resized = tf.image.resize(
                    generated_hsi, target_shape)
                combined_fake = tf.concat(
                    [generated_hsi_resized, augmented_rgb_batch_resized], axis=-1)
                gen_loss = generator_loss(discriminator(combined_fake))

            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))

            mse = mean_squared_error(hsi_batch, generated_hsi_resized)
            psnr = peak_signal_to_noise_ratio(hsi_batch, generated_hsi_resized)
            sam = spectral_angle_mapper(hsi_batch, generated_hsi_resized)

            """
            with summary_writer.as_default():
                tf.summary.scalar('Discriminator Loss', disc_loss, step=epoch *
                                  len(rgb_images) // config.BATCH_SIZE + i // config.BATCH_SIZE)
                tf.summary.scalar('Generator Loss', gen_loss, step=epoch *
                                  len(rgb_images) // config.BATCH_SIZE + i // config.BATCH_SIZE)
                tf.summary.scalar('MSE', mse, step=epoch * len(rgb_images) //
                                  config.BATCH_SIZE + i // config.BATCH_SIZE)
                tf.summary.scalar('PSNR', psnr, step=epoch * len(rgb_images) //
                                  config.BATCH_SIZE + i // config.BATCH_SIZE)
                tf.summary.scalar('SAM', sam, step=epoch * len(rgb_images) //
                                  config.BATCH_SIZE + i // config.BATCH_SIZE)
            """
            print(f'Epoch: {epoch}, Batch: {i // config.BATCH_SIZE}, Disc Loss: {disc_loss.numpy()}, Gen Loss: {gen_loss.numpy()}, MSE: {mse.numpy()}, PSNR: {psnr.numpy()}, SAM: {sam.numpy()}')
        checkpoint.save(file_prefix=checkpoint_path)


# Train the GAN
if __name__ == "__main__":
    # Data Augmentation
    # train_global()
    mode = "global"
    data_gen = ImageDataGenerator(rotation_range=20,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.1,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

    # Load data
    rgb_images = load_rgb_images(config.RGB_IMAGE_PATH)
    hsi_images = load_hsi_images_from_all_folders(config.HSI_IMAGE_PATH)

    # Model setup
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(
        config.LEARNING_RATE, beta_1=config.BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(
        config.LEARNING_RATE, beta_1=config.BETA_1)

    # Logging and Checkpointing
    log_dir = config.LOG_DIR
    summary_writer = tf.summary.create_file_writer(log_dir)
    # To save model checkpoints
    if mode == "global":
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, config.GLOBAL_CHECKPOINT_PREFIX)
    elif mode == "local":
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, config.LOCAL_CHECKPOINT_PREFIX)
    else:
        print("Error.")

    train_gan(rgb_images, hsi_images,  generator=generator,
              discriminator=discriminator, mode=mode)
