import time
import numpy as np
from IPython.display import display
from PIL import Image


def get_output(netG_alpha, netG_beta, X):
    real_input = X
    fake_output = netG_alpha.predict(real_input)
    rec_input = netG_beta.predict(fake_output)
    outputs = [fake_output, rec_input]
    return outputs


def get_combined_output(netG_alpha, netG_beta, X):
    r = [get_output(netG_alpha, netG_beta, X[i:i + 1]) for i in range(X.shape[0])]
    r = np.array(r)
    return r.swapaxes(0, 1)[:, :, 0]


def save_image(X, rows=1, image_size=256):
    assert X.shape[0] % rows == 0
    int_X = ((X * 255).clip(0, 255).astype('uint8'))
    int_X = int_X.reshape(-1, image_size, image_size, 3)
    int_X = int_X.reshape(rows, -1, image_size, image_size, 3).swapaxes(1, 2).reshape(rows * image_size, -1, 3)
    pil_X = Image.fromarray(int_X)
    t = str(time.time())
    pil_X.save(dpath + 'results/' + t, 'JPEG')


def show_generator_image(A, B, netG_alpha, netG_beta):
    assert A.shape == B.shape

    rA = get_combined_output(netG_alpha, netG_beta, A)
    rB = get_combined_output(netG_beta, netG_alpha, B)

    arr = np.concatenate([A, B, rA[0], rB[0], rA[1], rB[1]])
    save_image(arr, rows=3)
