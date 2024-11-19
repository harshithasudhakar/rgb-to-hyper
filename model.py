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
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=[vgg.get_layer('block3_conv3').output,
                     vgg.get_layer('block4_conv3').output]
        )
        self.feature_extractor.trainable = False

    def call(self, y_true, y_pred):
        # Preprocess inputs for VGG19
        y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
        y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

        # Extract features
        features_true = self.feature_extractor(y_true)
        features_pred = self.feature_extractor(y_pred)

        # Compute perceptual loss (e.g., MSE between features)
        loss = 0.0
        for ft, fp in zip(features_true, features_pred):
            loss += tf.reduce_mean(tf.square(ft - fp))
        return loss

# SAM Loss function
def spectral_angle_loss(y_true, y_pred):
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.norm(y_true, axis=-1)
    norm_pred = tf.norm(y_pred, axis=-1)
    cos_theta = dot_product / (norm_true * norm_pred + 1e-8)
    return tf.reduce_mean(tf.acos(tf.clip_by_value(cos_theta, -1.0, 1.0)))

# ResNet Block for Generator
class ResNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, name=None):
        super(ResNetBlock, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False, name=f'{name}_conv1')
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn1')
        self.relu = layers.Activation('relu', name=f'{name}_relu')
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False, name=f'{name}_conv2')
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn2')

    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return x + residual

class Generator(tf.keras.Model):
    def __init__(self, HSI_CHANNELS=31, name='generator'):
        super(Generator, self).__init__(name=name)
        # Encoder layers
        self.encoder1 = layers.Conv2D(64, (4, 4), strides=2, padding="same", use_bias=False, name='encoder1')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.2, name='leaky_relu')

        self.encoder2 = layers.Conv2D(128, (4, 4), strides=2, padding="same", use_bias=False, name='encoder2')
        self.bn2 = layers.BatchNormalization(name='bn2')

        self.encoder3 = layers.Conv2D(256, (4, 4), strides=2, padding="same", use_bias=False, name='encoder3')
        self.bn3 = layers.BatchNormalization(name='bn3')

        self.encoder4 = layers.Conv2D(512, (4, 4), strides=2, padding="same", use_bias=False, name='encoder4')
        self.bn4 = layers.BatchNormalization(name='bn4')

        # ResNet blocks
        self.resnet_blocks = [ResNetBlock(512, name=f'res_net_block_{i}') for i in range(6)]

        # Decoder layers
        self.decoder1 = layers.Conv2DTranspose(256, (4, 4), strides=2, padding="same", use_bias=False, name='decoder1')
        self.bn_decoder1 = layers.BatchNormalization(name='bn_decoder1')
        self.relu = layers.ReLU(name='relu')

        self.decoder2 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same", use_bias=False, name='decoder2')
        self.bn_decoder2 = layers.BatchNormalization(name='bn_decoder2')

        self.decoder3 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same", use_bias=False, name='decoder3')
        self.bn_decoder3 = layers.BatchNormalization(name='bn_decoder3')

        # Final decoder layer
        self.decoder4 = layers.Conv2DTranspose(
            HSI_CHANNELS, (4, 4), strides=2, padding="same", activation="tanh", use_bias=False, name='decoder4'
        )

        self.dropout = layers.Dropout(0.5, name='dropout')

        self.concat = layers.Concatenate(name='concat')

    def call(self, x, training=False):
        # Encoder
        e1 = self.encoder1(x)
        e1 = self.bn1(e1, training=training)
        e1 = self.leaky_relu(e1)

        e2 = self.encoder2(e1)
        e2 = self.bn2(e2, training=training)
        e2 = self.leaky_relu(e2)

        e3 = self.encoder3(e2)
        e3 = self.bn3(e3, training=training)
        e3 = self.leaky_relu(e3)

        e4 = self.encoder4(e3)
        e4 = self.bn4(e4, training=training)
        e4 = self.leaky_relu(e4)

        # ResNet blocks
        for block in self.resnet_blocks:
            e4 = block(e4, training=training)

        # Decoder
        d1 = self.decoder1(e4)
        d1 = self.bn_decoder1(d1, training=training)
        d1 = self.relu(d1)
        d1 = self.dropout(d1, training=training)
        d1 = self.concat([d1, e3])  # Skip connection

        d2 = self.decoder2(d1)
        d2 = self.bn_decoder2(d2, training=training)
        d2 = self.relu(d2)
        d2 = self.dropout(d2, training=training)
        d2 = self.concat([d2, e2])  # Skip connection

        d3 = self.decoder3(d2)
        d3 = self.bn_decoder3(d3, training=training)
        d3 = self.relu(d3)
        d3 = self.dropout(d3, training=training)
        d3 = self.concat([d3, e1])  # Skip connection

        # Final decoder layer to generate HSI
        d4 = self.decoder4(d3)

        return d4

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
