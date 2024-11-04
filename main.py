# Generates 31 .tiffs for each HSI (each .tiff corresponds to each channel in HSI)
import os
import numpy as np
import config
import tensorflow as tf
import imageio
from config import IMG_HEIGHT, IMG_WIDTH, RGB_IMAGE_PATH,HSI_IMAGE_PATH
from model import Generator, Discriminator
from loss import peak_signal_to_noise_ratio, spectral_angle_mapper, generator_loss, mean_squared_error, discriminator_loss
from utils import load_paired_images, visualize_generated_images, apply_paired_augmentation

def train_gan(rgb_path: str, hsi_path: str, generator: Generator,
              discriminator: Discriminator, target_size=(IMG_WIDTH, IMG_HEIGHT),
              mode="global"):
    """
    Train GAN with properly paired RGB and HSI images, using synchronized augmentation.
    """
    # Create metrics directory if it doesn't exist
    metrics_dir = './metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    # Create output directory for generated images if it doesn't exist
    generated_hsi_dir = r'C:\Harshi\ECS-II\Dataset\gen_hsi'
    os.makedirs(generated_hsi_dir, exist_ok=True)

    # Set up checkpointing based on mode
    if mode == "global":
        checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint('./checkpoints/'))
    else:
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )

    # Load paired images
    print("Loading and pairing images...")
    try:
        rgb_images, hsi_images = load_paired_images(rgb_path, hsi_path)
        print(f"Successfully loaded {len(rgb_images)} paired images")
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return

    # Convert to tensors
    rgb_images = tf.convert_to_tensor(rgb_images, dtype=tf.float32)
    hsi_images = tf.convert_to_tensor(hsi_images, dtype=tf.float32)

    # Dictionary to store final epoch metrics
    final_metrics = {
        'discriminator_loss': [],
        'generator_loss': [],
        'mse': [],
        'psnr': [],
        'sam': []
    }

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        for i in range(0, len(rgb_images), config.BATCH_SIZE):
            # Get batches
            rgb_batch = rgb_images[i:i + config.BATCH_SIZE]
            hsi_batch = hsi_images[i:i + config.BATCH_SIZE]

            # Apply synchronized augmentation
            try:
                augmented_rgb_batch, augmented_hsi_batch = apply_paired_augmentation(
                    rgb_batch, hsi_batch)
            except Exception as e:
                print(f"Error in data augmentation: {str(e)}")
                continue

            # Generator forward pass
            generated_hsi = generator(augmented_rgb_batch)

            # Visualize current results
            visualize_generated_images(
                augmented_rgb_batch, generated_hsi, augmented_hsi_batch,
                epoch, i // config.BATCH_SIZE)

            # Save generated HSI images
            for j in range(generated_hsi.shape[0]):
                # Convert tensor to numpy array and clip values if necessary
                generated_hsi_np = tf.clip_by_value(generated_hsi[j], 0, 1).numpy()
                
                # Normalize and ensure the data is in the correct format
                # Depending on your specific needs, you may want to apply different transformations.
                generated_hsi_np = np.moveaxis(generated_hsi_np, -1, 0)  # Move channels to first dimension
                
                # Ensure that the image is in a valid shape (C, H, W) for saving
                if generated_hsi_np.ndim == 3 and generated_hsi_np.shape[0] > 3:
                    # Save as a multi-page TIFF file if there are more than 3 channels
                    image_path = os.path.join(generated_hsi_dir, f'generated_hsi_epoch{epoch+1}_batch{i//config.BATCH_SIZE}_img{j}.tiff')
                    imageio.mimwrite(image_path, generated_hsi_np.astype(np.float32), format='tiff')
                else:
                    # Otherwise, save as normal RGB or grayscale
                    image_path = os.path.join(generated_hsi_dir, f'generated_hsi_epoch{epoch+1}_batch{i//config.BATCH_SIZE}_img{j}.png')
                    imageio.imwrite(image_path, (generated_hsi_np * 255).astype(np.uint8))  # Scale to 0-255 if saving as PNG

            # Prepare discriminator inputs
            combined_real = tf.concat([augmented_hsi_batch, augmented_rgb_batch], axis=-1)
            combined_fake = tf.concat([generated_hsi, augmented_rgb_batch], axis=-1)

            # Train discriminator
            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real)
                disc_fake = discriminator(combined_fake)
                disc_loss = discriminator_loss(disc_real, disc_fake)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Train generator
            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch)
                combined_fake = tf.concat([generated_hsi, augmented_rgb_batch], axis=-1)
                gen_loss = generator_loss(discriminator(combined_fake), generated_hsi, augmented_hsi_batch)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            # Calculate metrics
            mse = mean_squared_error(augmented_hsi_batch, generated_hsi)
            psnr = peak_signal_to_noise_ratio(augmented_hsi_batch, generated_hsi)
            sam = spectral_angle_mapper(augmented_hsi_batch, generated_hsi)

            # Store metrics for the final epoch
            if epoch == config.EPOCHS - 1:
                final_metrics['discriminator_loss'].append(disc_loss.numpy())
                final_metrics['generator_loss'].append(gen_loss.numpy())
                final_metrics['mse'].append(mse.numpy())
                final_metrics['psnr'].append(psnr.numpy())
                final_metrics['sam'].append(sam.numpy())

            # Print progress
            batch_idx = i // config.BATCH_SIZE
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Disc Loss: {disc_loss.numpy():.4f}, '
                  f'Gen Loss: {gen_loss.numpy():.4f}, '
                  f'MSE: {mse.numpy():.4f}, '
                  f'PSNR: {psnr.numpy():.4f}, '
                  f'SAM: {sam.numpy():.4f}')

        # Save checkpoint at end of epoch
        checkpoint.save(file_prefix=checkpoint_path)

        # Save metrics after the final epoch
        if epoch == config.EPOCHS - 1:
            metrics_file = os.path.join(metrics_dir, f'final_metrics_{mode}.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Training Mode: {mode}\n")
                f.write(f"Number of Batches: {len(final_metrics['mse'])}\n\n")
                f.write("Final Epoch Metrics (Average across all batches):\n")
                f.write("-" * 50 + "\n")
                f.write(
                    f"Discriminator Loss: {sum(final_metrics['discriminator_loss']) / len(final_metrics['discriminator_loss']):.4f}\n")
                f.write(
                    f"Generator Loss: {sum(final_metrics['generator_loss']) / len(final_metrics['generator_loss']):.4f}\n")
                f.write(
                    f"Mean Squared Error: {sum(final_metrics['mse']) / len(final_metrics['mse']):.4f}\n")
                f.write(
                    f"Peak Signal-to-Noise Ratio: {sum(final_metrics['psnr']) / len(final_metrics['psnr']):.4f}\n")
                f.write(
                    f"Spectral Angle Mapper: {sum(final_metrics['sam']) / len(final_metrics['sam']):.4f}\n")

# Train the GAN
if __name__ == "__main__":
    mode = "global"
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.BETA_1)

    # Logging and Checkpointing
    log_dir = config.LOG_DIR
    summary_writer = tf.summary.create_file_writer(log_dir)

    # To save model checkpoints
    if mode == "global":
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'ckpt')
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'local_ckpt')

    train_gan(rgb_path=RGB_IMAGE_PATH, hsi_path=HSI_IMAGE_PATH,
              generator=generator, discriminator=discriminator,
              mode=mode)
