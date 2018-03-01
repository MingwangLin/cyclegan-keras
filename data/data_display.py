import numpy as np
from IPython.display import display
from PIL import Image
from models.networks import cycle_generater


def display_image(X, rows=1, image_size=256):
    assert X.shape[0] % rows == 0
    int_X = ((X * 255).clip(0, 255).astype('uint8'))
    int_X = int_X.reshape(-1, image_size, image_size, 3)
    int_X = int_X.reshape(rows, -1, image_size, image_size, 3).swapaxes(1, 2).reshape(rows * image_size, -1, 3)
    display(Image.fromarray(int_X))


def show_generator_image(A, B, netG_A, netG_B):
    assert A.shape == B.shape

    def G(generater, X):
        r = np.array([generater([X[i:i + 1], 1]) for i in range(X.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]

    cycle_A_generater = cycle_generater(netG_A, netG_B)
    cycle_B_generater = cycle_generater(netG_B, netG_A)
    rA = G(cycle_A_generater, A)
    rB = G(cycle_B_generater, B)
    arr = np.concatenate([A, B, rA[0], rB[0], rA[1], rB[1]])
    display_image(arr, 3)
