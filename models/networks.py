import keras

from keras.layers import *

from keras_contrib.layers.normalization import InstanceNormalization


def conv_block(x, filters, size, stride=(2, 2), has_norm_layer=True, is_instance_layer=False,
               has_activation_layer=True, is_leaky_relu=False):
    x = Conv2D(filters, (size, size), strides=stride, padding=mode)(x)
    if has_norm_layer:
        if not is_instance_layer:
            x = BatchNormalization(axis=3)(x)
        else:
            x = InstanceNormalization(axis=3)(x)
    if has_activation_layer:
        if not is_leaky_relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.2)(x)
    return x


def res_block(ip, nf=256):
    x = ZeroPadding2D(padding=(1, 1))(ip)
    x = conv_block(x, nf, 3, (1, 1))
    Dropout(0.5)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, nf, 3, (1, 1), has_activation_layer=False)
    return add([x, ip])


def up_block(x, filters, size):
    x = UpSampling2D()(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters, (size, size), strides=(1, 1), padding='valid')(x)
    x = InstanceNormalization(axis=3)(x)
    return Activation('relu')(x)


def resnet_generator(arr):
    inp = Input(arr.shape[1:])
    x = ZeroPadding2D(padding=(3, 3))(inp)
    x = conv_block(x, 64, 7, (1, 1))
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 128, 3, (2, 2))
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 256, 3, (2, 2))
    for i in range(9):
        x = res_block(x)
    x = up_block(x, 128, 3)
    x = up_block(x, 64, 3)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(3, (7, 7), activation='tanh', strides=(1, 1), padding='valid')(x)
    outp = Lambda(lambda x: (x + 1) * 127.5)(x)
    return inp, outp


# Defines the PatchGAN discriminator
def NLayerDiscriminator(arr):
    inp = Input(arr.shape[1:])
    x = ZeroPadding2D(padding=(1, 1))(inp)
    x = conv_block(x, 64, 4, (2, 2), has_norm_layer=False, is_leaky_relu=True)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 128, 4, (2, 2), is_instance_layer=True, is_leaky_relu=True)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 256, 4, (2, 2), is_instance_layer=True, is_leaky_relu=True)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(1, (4, 4), activation='sigmoid', strides=(1, 1), padding='valid')(x)
    return inp, outp
