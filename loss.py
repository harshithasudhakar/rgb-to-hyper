import tensorflow as tf
from model import PerceptualLoss

def gradient_penalty(real, fake, discriminator):
    alpha = tf.random.uniform(shape=[real.shape[0], 1, 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def discriminator_loss(real_output, fake_output, real, fake, discriminator):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    gp = gradient_penalty(real, fake, discriminator)
    return tf.reduce_mean(real_loss + fake_loss) + 10.0 * gp  # Add gradient penalty


def generator_loss(fake_output, generated, target, lambda_pixel=10, lambda_perceptual=0.1):
    adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output))
    pixel_loss_value = pixel_loss(generated, target)
    perceptual_loss_value = PerceptualLoss()(target, generated)  # PerceptualLoss is used here
    return adv_loss + lambda_pixel * pixel_loss_value + lambda_perceptual * perceptual_loss_value


def pixel_loss(generated, target):
    return tf.reduce_mean(tf.square(generated - target))


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
