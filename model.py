# models.py
import tensorflow as tf
from tensorflow.keras import layers
from config import HSI_CHANNELS

# Generator Model (Subclassed Model)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')
        self.conv2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')
        self.bottleneck = layers.Conv2D(256, (4, 4), padding='same')
        self.deconv1 = layers.Conv2DTranspose(
            128, (4, 4), strides=2, padding='same')
        self.deconv2 = layers.Conv2DTranspose(
            64, (4, 4), strides=2, padding='same')
        self.output_layer = layers.Conv2D(
            HSI_CHANNELS, (3, 3), padding='same', activation='sigmoid')

    def call(self, inputs):
        x = tf.nn.relu(self.conv1(inputs))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.bottleneck(x))
        x = tf.nn.relu(self.deconv1(x))
        x = tf.nn.relu(self.deconv2(x))
        return self.output_layer(x)

# Discriminator Model (Subclassed Model)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')
        self.conv2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')
        self.conv3 = layers.Conv2D(256, (4, 4), padding='same')
        self.output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        x = tf.nn.leaky_relu(self.conv1(inputs), alpha=0.2)
        x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
        x = tf.nn.leaky_relu(self.conv3(x), alpha=0.2)
        return self.output_layer(x)
