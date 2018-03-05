import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import UpSampling2D, Conv2DTranspose, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras_contrib.layers.normalization import InstanceNormalization


def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=RandomNormal(0, 0.02), *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=3, epsilon=1e-5,
                              gamma_initializer=RandomNormal(1., 0.02))


def conv_block(x, filters, size, stride=(2, 2), has_norm_layer=True, use_norm_instance=False,
               has_activation_layer=True, use_leaky_relu=False, padding='same'):
    x = conv2d(filters, (size, size), strides=stride, padding=padding)(x)
    if has_norm_layer:
        if not use_norm_instance:
            x = batchnorm()(x)
        else:
            x = InstanceNormalization(axis=1)(x)
    if has_activation_layer:
        if not use_leaky_relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.2)(x)
    return x


def res_block(x, filters=256, use_dropout=False):
    y = conv_block(x, filters, 3, (1, 1))
    if use_dropout:
        y = Dropout(0.5)(y)
    y = conv_block(y, filters, 3, (1, 1), has_activation_layer=False)
    return Add()([y, x])


def up_block(x, filters, size, use_conv_transpose=True, use_norm_instance=False):
    if use_conv_transpose:
        x = Conv2DTranspose(filters, kernel_size=size, strides=2, padding='same',
                            use_bias=True if use_norm_instance else False,
                            kernel_initializer=RandomNormal(0, 0.02))(x)
        x = batchnorm()(x)
        x = Activation('relu')(x)

    else:
        x = UpSampling2D()(x)
        x = conv_block(x, filters, size, (1, 1))

    return x


# Defines the Resnet generator
def resnet_generator(image_size=256, input_nc=3, res_blocks=6):
    inputs = Input(shape=(image_size, image_size, input_nc))
    x = inputs

    x = conv_block(x, 64, 7, (1, 1))
    x = conv_block(x, 128, 3, (2, 2))
    x = conv_block(x, 256, 3, (2, 2))

    for i in range(res_blocks):
        x = res_block(x)

    x = up_block(x, 128, 3)
    x = up_block(x, 64, 3)

    x = conv2d(3, (7, 7), activation='tanh', strides=(1, 1), padding='same')(x)
    outputs = x

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


# Defines the PatchGAN discriminator
def n_layer_discriminator(image_size=256, input_nc=3, ndf=64, hidden_layers=3):
    """
        input_nc: input channels
        ndf: filters of the first layer
    """
    inputs = Input(shape=(image_size, image_size, input_nc))
    x = inputs

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, ndf, 4, has_norm_layer=False, use_leaky_relu=True, padding='valid')

    x = ZeroPadding2D(padding=(1, 1))(x)
    for i in range(1, hidden_layers + 1):
        nf = 2 ** i * ndf
        x = conv_block(x, nf, 4, use_leaky_relu=True, padding='valid')
        x = ZeroPadding2D(padding=(1, 1))(x)

    x = conv2d(1, (4, 4), activation='sigmoid', strides=(1, 1))(x)
    outputs = x

    return Model(inputs=[inputs], outputs=outputs), inputs, outputs

