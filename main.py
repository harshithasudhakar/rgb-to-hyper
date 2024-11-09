import os
import numpy as np
import tensorflow as tf
import imageio
import config
from config import IMG_HEIGHT, IMG_WIDTH, RGB_IMAGE_PATH, HSI_IMAGE_PATH
from model import Generator, Discriminator
from loss import peak_signal_to_noise_ratio, spectral_angle_mapper, generator_loss, mean_squared_error, discriminator_loss
from utils import load_paired_images, visualize_generated_images, apply_paired_augmentation


class GANLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, min_lr_ratio=0.1):
        super(GANLearningRateScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_lr = initial_learning_rate * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Warmup phase
        warmup_progress = tf.minimum(1.0, step / warmup_steps)
        warmup_lr = self.initial_learning_rate * warmup_progress

        # Decay phase with cosine decay
        decay_progress = tf.maximum(
            0.0, step - warmup_steps) / (decay_steps - warmup_steps)
        decay_factor = 0.5 * \
            (1.0 + tf.cos(tf.minimum(1.0, decay_progress) * np.pi))
        decay_lr = self.min_lr + \
            (self.initial_learning_rate - self.min_lr) * decay_factor

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)


def train_gan(rgb_path: str, hsi_path: str, generator: Generator,
              discriminator: Discriminator, generator_optimizer,
              discriminator_optimizer, summary_writer,
              target_size=(IMG_WIDTH, IMG_HEIGHT), mode="global"):
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
        checkpoint = tf.train.Checkpoint(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )
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
    steps = 0
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
            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch)
                combined_fake = tf.concat(
                    [generated_hsi, augmented_rgb_batch], axis=-1)
                gen_loss = generator_loss(discriminator(combined_fake),
                                          generated_hsi, augmented_hsi_batch)

            # Train generator
            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))

            # Prepare discriminator inputs
            combined_real = tf.concat(
                [augmented_hsi_batch, augmented_rgb_batch], axis=-1)

            # Train discriminator
            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real)
                disc_fake = discriminator(combined_fake)
                disc_loss = discriminator_loss(disc_real, disc_fake)

            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Calculate metrics
            mse = mean_squared_error(augmented_hsi_batch, generated_hsi)
            psnr = peak_signal_to_noise_ratio(
                augmented_hsi_batch, generated_hsi)
            sam = spectral_angle_mapper(augmented_hsi_batch, generated_hsi)

            # Log learning rates
            with summary_writer.as_default():
                gen_lr = generator_optimizer.learning_rate(steps)
                disc_lr = discriminator_optimizer.learning_rate(steps)
                tf.summary.scalar('generator_lr', gen_lr, step=steps)
                tf.summary.scalar('discriminator_lr', disc_lr, step=steps)
                tf.summary.scalar('generator_loss', gen_loss, step=steps)
                tf.summary.scalar('discriminator_loss', disc_loss, step=steps)
                tf.summary.scalar('mse', mse, step=steps)
                tf.summary.scalar('psnr', psnr, step=steps)
                tf.summary.scalar('sam', sam, step=steps)

            # Print progress every 10 steps
            if steps % 10 == 0:
                print(f'Epoch: {epoch}, Step: {steps}, '
                      f'Gen LR: {gen_lr.numpy():.6f}, '
                      f'Disc LR: {disc_lr.numpy():.6f}, '
                      f'Gen Loss: {gen_loss.numpy():.4f}, '
                      f'Disc Loss: {disc_loss.numpy():.4f}, '
                      f'PSNR: {psnr.numpy():.4f}')

            # Save generated images
            if steps % 100 == 0:
                # Save logic here (keeping your existing save logic)
                for j in range(generated_hsi.shape[0]):
                    generated_hsi_np = tf.clip_by_value(
                        generated_hsi[j], 0, 1).numpy()
                    generated_hsi_np = np.moveaxis(generated_hsi_np, -1, 0)

                    if generated_hsi_np.ndim == 3 and generated_hsi_np.shape[0] > 3:
                        image_path = os.path.join(
                            generated_hsi_dir,
                            f'generated_hsi_epoch{epoch+1}_step{steps}_img{j}.tiff')
                        imageio.mimwrite(image_path, generated_hsi_np.astype(
                            np.float32), format='tiff')
                    else:
                        image_path = os.path.join(
                            generated_hsi_dir,
                            f'generated_hsi_epoch{epoch+1}_step{steps}_img{j}.png')
                        imageio.imwrite(
                            image_path, (generated_hsi_np * 255).astype(np.uint8))

            # Store metrics for the final epoch
            if epoch == config.EPOCHS - 1:
                final_metrics['discriminator_loss'].append(disc_loss.numpy())
                final_metrics['generator_loss'].append(gen_loss.numpy())
                final_metrics['mse'].append(mse.numpy())
                final_metrics['psnr'].append(psnr.numpy())
                final_metrics['sam'].append(sam.numpy())

            steps += 1

        # Save checkpoint at end of epoch
        checkpoint.save(file_prefix=checkpoint_path)

        # Save final metrics
        if epoch == config.EPOCHS - 1:
            save_final_metrics(final_metrics, metrics_dir, mode)


def save_final_metrics(metrics, metrics_dir, mode):
    metrics_file = os.path.join(metrics_dir, f'final_metrics_{mode}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Training Mode: {mode}\n")
        f.write(f"Number of Batches: {len(metrics['mse'])}\n\n")
        f.write("Final Epoch Metrics (Average across all batches):\n")
        f.write("-" * 50 + "\n")
        for key in metrics:
            avg_value = sum(metrics[key]) / len(metrics[key])
            f.write(f"{key}: {avg_value:.4f}\n")


if __name__ == "__main__":
    mode = "global"
    generator = Generator()
    discriminator = Discriminator()

    # Calculate total steps for scheduler
    # You'll need to add dataset size to your config
    steps_per_epoch = config.DATASET_SIZE // config.BATCH_SIZE
    total_steps = config.EPOCHS * steps_per_epoch

    # Setup schedulers
    gen_scheduler = GANLearningRateScheduler(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=total_steps,
        warmup_steps=total_steps // 20,  # 5% of total steps for warmup
        min_lr_ratio=0.1
    )

    disc_scheduler = GANLearningRateScheduler(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=total_steps,
        warmup_steps=total_steps // 20,
        min_lr_ratio=0.05
    )

    # Create optimizers with schedulers
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=gen_scheduler,
        beta_1=config.BETA_1
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=disc_scheduler,
        beta_1=config.BETA_1
    )

    # Logging and Checkpointing
    log_dir = config.LOG_DIR
    summary_writer = tf.summary.create_file_writer(log_dir)

    # To save model checkpoints
    if mode == "global":
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'ckpt')
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'local_ckpt')

    train_gan(
        rgb_path=RGB_IMAGE_PATH,
        hsi_path=HSI_IMAGE_PATH,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        summary_writer=summary_writer,
        mode=mode
    )
