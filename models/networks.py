import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import UpSampling2D, Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import add
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras_contrib.layers.normalization import InstanceNormalization


def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=RandomNormal(0, 0.02), *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=3, epsilon=1.01e-5,
                              gamma_initializer=RandomNormal(1., 0.02))


def conv_block(x, nf, size, stride=(2, 2), use_norm_layer=True, use_norm_instance=False,
               use_activation_function=True, use_leaky_relu=False):
    x = conv2d(nf, (size, size), strides=stride)(x)
    if use_norm_layer:
        if not use_norm_instance:
            x = batchnorm()(x, training=1)
        else:
            x = InstanceNormalization(axis=1)(x)
    if use_activation_function:
        if not use_leaky_relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.2)(x)
    return x


def res_block(ip, nf=256):
    x = ZeroPadding2D(padding=(1, 1))(ip)
    x = conv_block(x, nf, 3, (1, 1))
    Dropout(0.5)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, nf, 3, (1, 1), use_activation_function=False)
    return add([x, ip])


def up_block(x, nf, size, use_conv_transpose=False, use_norm_instance=False):
    if use_conv_transpose:
        x = Conv2DTranspose(nf, kernel_size=size, strides=2, padding='same',
                            use_bias=True if use_norm_instance else False,
                            kernel_initializer=RandomNormal(0, 0.02))(x)
    else:
        x = UpSampling2D()(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = conv2d(nf, (size, size), strides=(1, 1))(x)

    x = batchnorm()(x, training=1)
    x = Activation('relu')(x)

    return x


def resnet_generator(img_size=256, input_nc=3, res_blocks=9):
    inputs = Input(shape=(img_size, img_size, input_nc))
    x = inputs
    print('x1', x.shape)
    x = ZeroPadding2D(padding=(3, 3))(x)
    print('x2', x.shape)
    x = conv_block(x, 64, 7, (1, 1))
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 128, 3, (2, 2))
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 256, 3, (2, 2))
    for i in range(res_blocks):
        x = res_block(x)
    x = up_block(x, 128, 3)
    x = up_block(x, 64, 3)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = conv2d(3, (7, 7), activation='tanh', strides=(1, 1))(x)
    outputs = x
    return Model(inputs=inputs, outputs=[outputs])


# Defines the PatchGAN discriminator
def n_layer_discriminator(input_nc=3, ndf=64, hidden_layers=2):
    """
        input_nc: input channels
        ndf: filters of the first layer
    """
    inputs = Input(shape=(None, None, input_nc))
    print('inputs', inputs.shape)
    x = inputs
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, ndf, 4, use_norm_layer=False, use_leaky_relu=True)
    x = ZeroPadding2D(padding=(1, 1))(x)
    for i in range(1, hidden_layers + 1):
        nf = 2 ** i * ndf
        x = conv_block(x, nf, 4, use_leaky_relu=True)
        x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv2d(1, (4, 4), activation='sigmoid', strides=(1, 1))(x)
    outputs = x
    print('outputs', outputs.shape)
    return Model(inputs=inputs, outputs=outputs)
