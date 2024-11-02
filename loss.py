import tensorflow as tf


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_output), fake_output)
    return tf.reduce_mean(real_loss + fake_loss)


def generator_loss(fake_output, generated, target, lambda_pixel=10):
    adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output))
    pixel_loss_value = pixel_loss(generated, target)
    return adv_loss + lambda_pixel * pixel_loss_value


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
