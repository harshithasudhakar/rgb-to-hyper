import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from config import HSI_CHANNELS

# Spectral Normalization Layer
class SpectralNormalization(layers.Layer):
    def __init__(self, layer, power_iterations=1):
        super(SpectralNormalization, self).__init__()
        self.layer = layer
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(shape=(1, self.w.shape[-1]), initializer="random_normal", trainable=False, name="sn_u")

    def call(self, inputs, training=None):
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped))
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        self.u.assign(u)
        w_sn = self.w / sigma
        
        # Apply convolution with normalized weights
        outputs = tf.nn.conv2d(inputs, w_sn, strides=self.layer.strides, padding=self.layer.padding.upper())
        
        if self.layer.use_bias:
            outputs = tf.nn.bias_add(outputs, self.layer.bias)
        return outputs

# Perceptual Loss using VGG19
class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = VGG19(include_top=False, weights='imagenet')
        self.feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        self.feature_extractor.trainable = False

    def call(self, y_true, y_pred):
        # Convert HSI to RGB by averaging or selecting specific bands
        y_true_rgb = tf.image.resize(y_true[..., :3], (224, 224))  # Resize to VGG19 input size
        y_pred_rgb = tf.image.resize(y_pred[..., :3], (224, 224))  # Resize to VGG19 input size

        y_true_features = self.feature_extractor(y_true_rgb)
        y_pred_features = self.feature_extractor(y_pred_rgb)
        
        # Calculate perceptual loss using the extracted features
        perceptual_loss_value = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
        
        # Calculate pixel-wise loss using the original HSI images
        pixel_loss_value = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Combine perceptual loss and pixel-wise loss
        return perceptual_loss_value + pixel_loss_value

# SAM Loss function
def spectral_angle_loss(y_true, y_pred):
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.norm(y_true, axis=-1)
    norm_pred = tf.norm(y_pred, axis=-1)
    cos_theta = dot_product / (norm_true * norm_pred + 1e-8)
    return tf.reduce_mean(tf.acos(tf.clip_by_value(cos_theta, -1.0, 1.0)))

# ResNet Block for Generator
class ResNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.in1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.in2 = layers.BatchNormalization()

    def call(self, x):
        residual = x
        x = tf.nn.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        return x + residual

# Generator Model
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = layers.Conv2D(128, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.encoder2 = layers.Conv2D(256, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.encoder3 = layers.Conv2D(512, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.encoder4 = layers.Conv2D(1024, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.encoder5 = layers.Conv2D(2048, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.resnet_blocks = [ResNetBlock(2048) for _ in range(6)]  # Increase filters
        self.decoder1 = layers.Conv2DTranspose(1024, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.decoder2 = layers.Conv2DTranspose(512, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.decoder3 = layers.Conv2DTranspose(256, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.decoder4 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same", use_bias=False)  # Increase filters
        self.decoder5 = layers.Conv2DTranspose(HSI_CHANNELS, (4, 4), strides=2, padding="same", activation="tanh", use_bias=False)
        self.dropout = layers.Dropout(0.5)  # Add dropout

    def call(self, x, training=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(tf.nn.leaky_relu(e1))
        e3 = self.encoder3(tf.nn.leaky_relu(e2))
        e4 = self.encoder4(tf.nn.leaky_relu(e3))
        e5 = self.encoder5(tf.nn.leaky_relu(e4))
        for block in self.resnet_blocks:
            e5 = block(e5)
        d1 = self.decoder1(tf.nn.relu(e5))
        d1 = layers.Concatenate()([d1, e4])  # Skip connection
        d2 = self.decoder2(tf.nn.relu(d1))
        d2 = layers.Concatenate()([d2, e3])  # Skip connection
        d3 = self.decoder3(tf.nn.relu(d2))
        d3 = layers.Concatenate()([d3, e2])  # Skip connection
        d4 = self.decoder4(tf.nn.relu(d3))
        d4 = layers.Concatenate()([d4, e1])  # Skip connection
        d5 = self.decoder5(tf.nn.relu(d4))
        if training:
            d5 = self.dropout(d5, training=training)  # Apply dropout during training
        return d5

# Discriminator Model
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            SpectralNormalization(layers.Conv2D(64, (4, 4), strides=2, padding="same")),
            layers.LeakyReLU(),
            SpectralNormalization(layers.Conv2D(128, (4, 4), strides=2, padding="same")),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            SpectralNormalization(layers.Conv2D(256, (4, 4), strides=2, padding="same")),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            SpectralNormalization(layers.Conv2D(512, (4, 4), strides=1, padding="same")),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            SpectralNormalization(layers.Conv2D(1, (4, 4), strides=1, padding="same")),
            layers.Activation('sigmoid')
        ])

    def call(self, x):
        return self.model(x)
