import numpy as np
import glob
from PIL import Image
from random import randint, shuffle


def load_data(file_pattern):
    return glob.glob(file_pattern)


def read_image(img, loadsize=286, imagesize=256):
    img = Image.open(img).convert('RGB')
    img = img.resize((loadsize, loadsize), Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255
    # random jitter
    w_offset = h_offset = randint(0, max(0, loadsize - imagesize - 1))
    img = img[h_offset:h_offset + imagesize,
          w_offset:w_offset + imagesize, :]
    # horizontal flip
    if randint(0, 1):
        img = img[:, ::-1]
    return img


def try_read_img(data, index):
    try:
        img = read_image(data[index])
        return img
    except:
        try_read_img(data, index + 1)


def minibatch(data, batch_size):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batch_size
        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = []
        for j in range(i, i + size):
            img = try_read_img(data, j)
            rtn.append(img)

        i += size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, dataB, batch_size):
    batchA = minibatch(dataA, batch_size)
    batchB = minibatch(dataB, batch_size)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B
