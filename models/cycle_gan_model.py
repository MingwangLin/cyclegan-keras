import torch
import os
import itertools
import sys
import util.util as util
import numpy as np
import keras.backend as K
from collections import OrderedDict
from .networks import resnet_generator, n_layer_discriminator
from .base_model import BaseModel
from keras.optimizers import Adam


class KerasCycleGAN:
    def name(self):
        return 'KerasCycleGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.G_B = resnet_generator()
        self.G_A = resnet_generator()

        self.D_B = n_layer_discriminator()
        self.D_A = n_layer_discriminator()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_A = input_A.numpy().astype('float32')
        input_A = np.moveaxis(input_A, 1, -1)
        self.real_A = input_A
        input_B = input['B' if AtoB else 'A']
        input_B = input_B.numpy().astype('float32')
        input_B = np.moveaxis(input_B, 1, -1)
        self.real_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        fake_B = self.netG_A.predict(self.real_A)
        self.rec_A = self.netG_B.predict(fake_B)

        fake_A = self.netG_B.predict(self.real_B)
        self.rec_B = self.netG_A.predict(fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def loss_function_basic(self, output, target):
        epsilon = 1e-12
        result = -K.mean(K.log(output + epsilon) * target + K.log(1 - output + epsilon) * (1 - target))
        return result

    def loss_function_lsgan(self, output, target):
        result = K.mean(K.abs(K.square(output - target)))
        return result

    def loss(self, net_D, real, fake, rec):
        # output_real = net_D([real])
        # output_fake = net_D([fake])
        output_real = net_D.predict([real])
        output_fake = net_D.predict([fake])
        # GAN loss D
        loss_D_real = self.loss_function_lsgan(output_real, K.ones_like(output_real))
        loss_D_fake = self.loss_function_lsgan(output_fake, K.zeros_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        # GAN loss G
        loss_G = self.loss_function_lsgan(output_fake, K.ones_like(output_fake))
        # cycle loss
        loss_cyc = K.mean(K.abs(rec - real))

        return loss_D, loss_G, loss_cyc

    def G_A_forward(self, G_A, G_B):
        # real_input = G_A.input[0]
        # fake_output = G_A.output[0]
        # rec_input = G_B([fake_output])
        real_input = self.real_A
        print('-----G_A_forward-----')
        print('-----real_input-----', real_input.shape)
        fake_output = G_B.predict(self.real_A)
        print('-----fake_output-----', fake_output.shape)
        rec_input = G_B.predict([fake_output])
        print('-----rec_input-----', rec_input.shape)
        return real_input, fake_output, rec_input

    def G_B_forward(self, G_B, G_A):
        # real_input = G_B.input[0]
        # fake_output = G_B.output[0]
        # rec_input = G_A([fake_output])
        real_input = self.real_B
        print('-----G_B_forward-----')
        print('-----real_input-----', real_input.shape)
        # fake_output = G_A.output[0]
        fake_output = G_B.predict(self.real_B)
        print('-----fake_output-----', fake_output.shape)
        rec_input = G_B.predict([fake_output])
        print('-----rec_input-----', rec_input.shape)
        return real_input, fake_output, rec_input

    def backward(self):
        real_A, fake_B, rec_A = self.G_A_forward(self.G_A, self.G_B)
        real_B, fake_A, rec_B = self.G_B_forward(self.G_B, self.G_A)

        loss_DA, loss_GA, loss_cycA = self.loss(self.D_A, real_A, fake_A, rec_A)
        loss_DB, loss_GB, loss_cycB = self.loss(self.D_B, real_B, fake_B, rec_B)
        loss_D = loss_DA + loss_DB

        loss_cyc = loss_cycA + loss_cycB
        lambda_param = self.opt.lambda_param
        loss_G = loss_GA + loss_GB + lambda_param * loss_cyc

        weights_D = self.D_A.trainable_weights + self.D_B.trainable_weights
        weights_G = self.G_A.trainable_weights + self.G_B.trainable_weights
        print('-----beta1-----', self.opt.beta1)
        print('-----weights_D-----', weights_D, weights_G)
        print('-----lr-----', self.opt.lr)
        print('-----loss_D-----', loss_DA, loss_DB)
        training_updates = Adam(lr=self.opt.lr, beta_1=self.opt.beta1).get_updates(weights_D, [], loss_D)
        D_backward = K.function([real_A, real_B], [loss_DA / 2, loss_DB / 2], training_updates)

        training_updates = Adam(lr=self.opt.lr, beta_1=self.opt.beta1).get_updates(weights_G, [], loss_G)
        G_backward = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)

        return G_backward, D_backward

    def optimize_parameters(self):
        G_backward, D_backward = self.backward()
        print('-----hit-----')
        self.loss_G_A, self.loss_G_B, loss_cycle = G_backward([self.real_A, self.real_B])
        self.loss_D_A, self.loss_D_B = D_backward([self.real_A, self.real_B])

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc', self.loss_cycle),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B)])
        return ret_errors

    def save(self, epoch_count, iter_count):
        save_filename = '{}_{}_GA.hdf5'.format(epoch_count, iter_count)
        save_path = os.path.join(self.save_dir, save_filename)
        self.G_A.save(save_path)

        save_filename = '{}_{}_GA.json'.format(epoch_count, iter_count)
        save_path = os.path.join(self.save_dir, save_filename)
        with open(save_path, 'w') as f:
            f.write(self.G_A.to_json())

        save_filename = '{}_{}_GA_weights.hdf5'.format(epoch_count, iter_count)
        save_path = os.path.join(self.save_dir, save_filename)
        self.G_A.save_weights(save_path)

        save_filename = '{}_{}_GA_weights.json'.format(epoch_count, iter_count)
        save_path = os.path.join(self.save_dir, save_filename)
        with open(save_path, 'w') as f:
            f.write(self.G_A.to_json())
