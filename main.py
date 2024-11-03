import os
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy
import config
from config import IMG_HEIGHT, IMG_WIDTH
from model import Generator, Discriminator
from loss import peak_signal_to_noise_ratio, spectral_angle_mapper, generator_loss, mean_squared_error, discriminator_loss
from utils import load_paired_images, visualize_generated_images, apply_paired_augmentation


def blind_deconvolution(image, max_iter=30):
    """
    Apply blind deconvolution using Richardson-Lucy algorithm
    """
    # Convert to numpy if tensor
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Estimate initial PSF (Point Spread Function)
    psf_size = 5
    psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)

    # Apply Richardson-Lucy deconvolution
    deblurred = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        deblurred[..., channel] = richardson_lucy(
            image[..., channel], psf, num_iter=max_iter)

    return deblurred


def save_generated_images(images, save_dir, prefix):
    """
    Save images as numpy arrays
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, img in enumerate(images):
        # Convert to numpy and ensure proper scaling
        img_np = img.numpy() if isinstance(img, tf.Tensor) else img
        img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0,1]
        np.save(os.path.join(save_dir, f'{prefix}_img_{idx}.npy'), img_np)


def train_gan(rgb_path: str, hsi_path: str, generator: Generator,
              discriminator: Discriminator, target_size=(IMG_WIDTH, IMG_HEIGHT),
              mode="global", lambda_mae=0.1):
    """
    Train GAN with properly paired RGB and HSI images, using synchronized augmentation.
    Added MAE loss to generator loss computation.
    """
    # Create metrics directory if it doesn't exist
    metrics_dir = './metrics'
    generated_dir = './data/generated'
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    # Set up checkpointing based on mode
    if mode == "global":
        checkpoint = tf.train.Checkpoint(
            generator=generator, discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint('./checkpoints/'))
    else:
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator)

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

    final_generated_images = []
    final_real_images = []

    # Dictionary to store final epoch metrics
    final_metrics = {
        'discriminator_loss': [],
        'generator_loss': [],
        'mae_loss': [],  # Added MAE tracking
        'mse': [],
        'psnr': [],
        'sam': []
    }

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        is_final_epoch = epoch == config.EPOCHS - 1

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

            if is_final_epoch:
                final_generated_images.extend(generated_hsi)
                final_real_images.extend(augmented_hsi_batch)

            # Visualize current results
            """
            visualize_generated_images(
                augmented_rgb_batch, generated_hsi, augmented_hsi_batch,
                epoch, i // config.BATCH_SIZE)
            """

            # Prepare discriminator inputs
            combined_real = tf.concat(
                [augmented_hsi_batch, augmented_rgb_batch], axis=-1)
            combined_fake = tf.concat(
                [generated_hsi, augmented_rgb_batch], axis=-1)

            # Train discriminator
            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real)
                disc_fake = discriminator(combined_fake)
                disc_loss = discriminator_loss(disc_real, disc_fake)

            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Train generator with added MAE loss
            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch)
                combined_fake = tf.concat(
                    [generated_hsi, augmented_rgb_batch], axis=-1)

                # Calculate original generator loss
                gen_loss_original = generator_loss(discriminator(
                    combined_fake), generated_hsi, augmented_hsi_batch)

                # Calculate MAE loss
                mae_loss = tf.reduce_mean(
                    tf.abs(generated_hsi - augmented_hsi_batch))

                # Combine losses with weighting factor
                gen_loss = gen_loss_original + lambda_mae * mae_loss

            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))

            # Calculate metrics
            mse = mean_squared_error(augmented_hsi_batch, generated_hsi)
            psnr = peak_signal_to_noise_ratio(
                augmented_hsi_batch, generated_hsi)
            sam = spectral_angle_mapper(augmented_hsi_batch, generated_hsi)

            # Store metrics for the final epoch
            if epoch == config.EPOCHS - 1:
                final_metrics['discriminator_loss'].append(disc_loss.numpy())
                final_metrics['generator_loss'].append(gen_loss.numpy())
                final_metrics['mae_loss'].append(mae_loss.numpy())  # Track MAE
                final_metrics['mse'].append(mse.numpy())
                final_metrics['psnr'].append(psnr.numpy())
                final_metrics['sam'].append(sam.numpy())

            # Print progress with added MAE loss
            batch_idx = i // config.BATCH_SIZE
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Disc Loss: {disc_loss.numpy():.4f}, '
                  f'Gen Loss: {gen_loss.numpy():.4f}, '
                  f'MAE Loss: {mae_loss.numpy():.4f}, '
                  f'MSE: {mse.numpy():.4f}, '
                  f'PSNR: {psnr.numpy():.4f}, '
                  f'SAM: {sam.numpy():.4f}')

        # Save checkpoint at end of epoch
        checkpoint.save(file_prefix=checkpoint_path)

        # Save metrics after the final epoch with added MAE
        if is_final_epoch:
            # Save generated images
            save_generated_images(final_generated_images,
                                  generated_dir, 'generated')
            save_generated_images(final_real_images, generated_dir, 'real')

            # Save metrics
            metrics_file = os.path.join(
                metrics_dir, f'final_metrics_{mode}.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Training Mode: {mode}\n")
                f.write(f"Number of Batches: {len(final_metrics['mse'])}\n\n")
                f.write("Final Epoch Metrics (Average across all batches):\n")
                f.write("-" * 50 + "\n")
                for metric_name, values in final_metrics.items():
                    avg_value = sum(values) / len(values)
                    f.write(f"{metric_name}: {avg_value:.4f}\n")

    return final_generated_images, final_real_images


# Rest of the code remains the same
if __name__ == "__main__":
    mode = "global"
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
        exit(1)

    generated_images, real_images = train_gan(config.RGB_IMAGE_PATH, config.HSI_IMAGE_PATH, generator=generator,
                                              discriminator=discriminator, mode=mode)

    if generated_images is not None and real_images is not None:
        print("\nApplying deblurring to generated images...")

        # Create directory for deblurred images
        deblurred_dir = './deblurred_images'
        os.makedirs(deblurred_dir, exist_ok=True)

        # Apply deblurring and calculate metrics
        deblurred_images = []
        original_psnr_values = []
        deblurred_psnr_values = []

        for gen_img, real_img in zip(generated_images, real_images):
            # Calculate PSNR before deblurring
            original_psnr = peak_signal_to_noise_ratio(real_img, gen_img)
            original_psnr_values.append(original_psnr)

            # Apply deblurring
            deblurred_img = blind_deconvolution(gen_img)
            deblurred_images.append(deblurred_img)

            # Calculate PSNR after deblurring
            deblurred_psnr = peak_signal_to_noise_ratio(
                real_img, tf.convert_to_tensor(deblurred_img))
            deblurred_psnr_values.append(deblurred_psnr)

        # Save deblurred images
        save_generated_images(deblurred_images, deblurred_dir, 'deblurred')

        # Save deblurring metrics
        deblur_metrics_file = os.path.join(
            './metrics', 'deblurring_metrics.txt')
        with open(deblur_metrics_file, 'w') as f:
            f.write("Deblurring Results:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"Average PSNR before deblurring: {np.mean(original_psnr_values):.4f}\n")
            f.write(
                f"Average PSNR after deblurring: {np.mean(deblurred_psnr_values):.4f}\n")
            f.write(
                f"PSNR improvement: {np.mean(deblurred_psnr_values) - np.mean(original_psnr_values):.4f}\n")

        print("Deblurring process completed. Results saved to:", deblurred_dir)
    else:
        print("Training failed or no images were generated.")
