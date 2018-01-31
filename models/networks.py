import keras

from keras.layers import *

from keras_contrib.layers.normalization import InstanceNormalization



def conv_block(x, filters, size, stride=(2, 2), mode='same', act=True):
    x = Conv2D(filters, (size, size), strides=stride, padding=mode)(x)
    x = InstanceNormalization(axis=3)(x)
    return Activation('relu')(x) if act else x


def res_block(ip, nf=256):
    x = ZeroPadding2D(padding=(1, 1))(ip)
    x = conv_block(x, nf, 3, (1, 1), mode='valid')
    Dropout(0.5)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, nf, 3, (1, 1), act=False, mode='valid')
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
    x = conv_block(x, 64, 7, (1, 1), mode='valid')
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 128, 3, (2, 2), mode='valid')
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv_block(x, 256, 3, (2, 2), mode='valid')
    for i in range(9):
        x = res_block(x)
    x = up_block(x, 128, 3)
    x = up_block(x, 64, 3)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(3, (7, 7), activation='tanh', strides=(1, 1), padding='valid')(x)
    outp = Lambda(lambda x: (x + 1) * 127.5)(x)
    return inp, outp
