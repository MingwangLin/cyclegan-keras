# cyclegan-keras

keras implementation of cycle-gan based on [pytorch-CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (by junyanz) and <a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei)

## Prerequisites
train.py has not been tested, CycleGAN-keras.ipynb is recommended and tested OK on
- Ubuntu 16.04
- Python 3.6
- Keras 2.1.2
- Tensorflow 1.0.1
- NVIDIA GPU + CUDA8.0 CuDNN6 or CuDNN5



## Demos [[manga-colorization-demo]](http://www.styletransfer.tech) 

Colorize manga with Cycle-GAN model totally run in browser.
- Built based on [Keras.js](https://github.com/transcranial/keras-js) and [keras.js demos](https://transcranial.github.io/keras-js)
- Model trained by juyter notebook version of this git repo
- Check [Demo-Introduction](https://zhuanlan.zhihu.com/p/34672860) or my [demo-repo](https://github.com/MingwangLin/manga-colorization) for more details


