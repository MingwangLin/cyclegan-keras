import numpy as np
from IPython.display import display
from PIL import Image
from models.loss import get_model_variables


def show_img(img_batch, rows=1, imagesize=256):
    assert img_batch.shape[0] % rows == 0
    img_batch = ((img_batch * 255).clip(0, 255).astype('uint8'))
    img_batch = img_batch.reshape(-1, imagesize, imagesize, 3)
    img_batch = img_batch.reshape(rows, -1, imagesize, imagesize, 3).swapaxes(1, 2).reshape(rows * imagesize, -1, 3)
    display(Image.fromarray(img_batch))


def show_model_img(A, B, netGA, netGB):
    assert A.shape == B.shape

    def G(compute_output, X):
        r = np.array([compute_output([X[i:i + 1], 1]) for i in range(X.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]

    _, _, _, compute_GA_output = get_model_variables(netGA, netGB)
    _, _, _, compute_GB_output = get_model_variables(netGB, netGA)
    rA = G(compute_GA_output, A)
    rB = G(compute_GB_output, B)
    arr = np.concatenate([A, B, rA[0], rB[0], rA[1], rB[1]])
    show_img(arr, 3)
