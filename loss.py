import tensorflow as tf
from model import PerceptualLoss

def gradient_penalty(real, fake, discriminator):
    """
    Calculate the gradient penalty for WGAN-GP.

    Parameters:
    - real: Concatenated real RGB and HSI tensors. Shape: (batch_size, H, W, 34)
    - fake: Concatenated fake RGB and HSI tensors. Shape: (batch_size, H, W, 34)
    - discriminator: The Discriminator model instance.

    Returns:
    - Gradient penalty scalar.
    """
    batch_size = tf.shape(real)[0]
    # Sample random numbers for interpolation
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    
    # Create interpolated samples
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    # Compute gradients w.r.t. the interpolated samples
    gradients = gp_tape.gradient(pred, interpolated)
    gradients = tf.reshape(gradients, [batch_size, -1])
    
    # Compute L2 norm of gradients for each sample
    gradient_norm = tf.norm(gradients, axis=1)
    
    # Compute gradient penalty
    gp = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return gp

def discriminator_loss(real_inputs, fake_inputs, discriminator, lambda_gp=10.0):
    """
    Calculate the Discriminator loss with Gradient Penalty.

    Parameters:
    - real_inputs: Concatenated real RGB and HSI tensors. Shape: (batch_size, H, W, 34)
    - fake_inputs: Concatenated fake RGB and HSI tensors. Shape: (batch_size, H, W, 34)
    - discriminator: The Discriminator model instance.
    - lambda_gp: Weight for the gradient penalty term.

    Returns:
    - Total Discriminator loss.
    """
    # Get Discriminator outputs
    real_output = discriminator(real_inputs, training=True)
    fake_output = discriminator(fake_inputs, training=True)

    # Calculate real and fake losses
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    
    # Calculate Gradient Penalty
    gp = gradient_penalty(real_inputs, fake_inputs, discriminator)
    
    # Total Discriminator loss
    total_loss = fake_loss - real_loss + lambda_gp * gp
    return total_loss


def generator_loss(fake_output, generated, target, lambda_pixel=1.0, lambda_perceptual=0.5):
    adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output))
    pixel_loss_value = pixel_loss(generated, target)
    perceptual_loss_value = PerceptualLoss()(target, generated)  # PerceptualLoss is used here
    return adv_loss + lambda_pixel * pixel_loss_value + lambda_perceptual * perceptual_loss_value


def pixel_loss(generated, target):
    return tf.reduce_mean(tf.abs(generated - target))


def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def spectral_angle_mapper(y_true, y_pred, epsilon=1e-7):
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    y_true_normalized = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)

    cos_theta = tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=-1)
    cos_theta = tf.clip_by_value(cos_theta, -1.0 + epsilon, 1.0 - epsilon)

    return tf.reduce_mean(tf.acos(cos_theta))
