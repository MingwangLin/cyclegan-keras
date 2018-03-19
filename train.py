import time
import numpy as np
from IPython.display import clear_output
from options.train_options import TrainOptions
from data.data_loader import load_data, minibatchAB
from data.save_data import show_generator_image
from util.image_pool import ImagePool
from models.networks import get_generater_function
from models.networks import resnet_generator, n_layer_discriminator
from models.train_function import *
opt = TrainOptions().parse()

# load data
dpath = opt.dataroot
train_A = load_data(dpath + 'trainA/*')
train_B = load_data(dpath + 'trainB/*')
train_batch = minibatchAB(train_A, train_B, batch_size=opt.batch_size)
val_A = load_data(dpath + 'valA/*')
val_B = load_data(dpath + 'valB/*')
val_batch = minibatchAB(val_A, val_B, batch_size=4)

# create gennerator models
netG_A, real_A, fake_B = resnet_generator()
netG_B, real_B, fake_A = resnet_generator()

# create discriminator models
netD_A = n_layer_discriminator()
netD_B = n_layer_discriminator()

# create generators train function
netG_train_function = netG_train_function_creator(netD_A, netD_B, netG_A, netG_B, real_A, real_B, fake_A, fake_B)
# create discriminator A train function
netD_A_train_function = netD_A_train_function(netD_A, netD_B, netG_A, netG_B, real_A, opt.finesize, opt.input_nc)
# create discriminator B train function
netD_B_train_function = netD_A_train_function(netD_A, netD_B, netG_A, netG_B, real_B, opt.finesize, opt.input_nc)

# train loop
time_start = time.time()
how_many_epochs = 5
iteration_count = 0
epoch_count = 0
batch_size = opt.batch_size
display_freq = 10000

netG_A_function = get_generater_function(netG_A)
netG_B_functionr = get_generater_function(netG_B)

fake_A_pool = ImagePool()
fake_B_pool = ImagePool()

while epoch_count < how_many_epochs:
    target_label = np.zeros((batch_size, 1))
    epoch_count, A, B = next(train_batch)

    tmp_fake_B = netG_A_function([A])[0]
    tmp_fake_A = netG_B_functionr([B])[0]

    _fake_B = fake_B_pool.query(tmp_fake_B)
    _fake_A = fake_A_pool.query(tmp_fake_A)

    netG_train_function.train_on_batch([A, B], target_label)

    netD_B_train_function.train_on_batch([B, _fake_B], target_label)
    netD_A_train_function.train_on_batch([A, _fake_A], target_label)

    iteration_count += 1

    if iteration_count % display_freq == 0:
        clear_output()
        traintime = (time.time() - time_start) / iteration_count
        print('epoch_count: {}  iter_count: {}  timecost/iter: {}s'.format(epoch_count, iteration_count, traintime))
        _, val_A, val_B = next(val_batch)
        show_generator_image(val_A, val_B, netG_A, netG_B)

        save_name = dpath + '{}' + str(iteration_count) + '.h5'

        netG_A.save(save_name.format('tf_GA'))
        netG_A.save_weights(save_name.format('tf_GA_weights'))
        netG_B.save(save_name.format('tf_GB'))
        netG_B.save_weights(save_name.format('tf_GB_weights'))
        netD_A.save(save_name.format('tf_DA'))

        netG_train_function.save(save_name.format('tf_G_train'))
        netG_train_function.save_weights(save_name.format('tf_G_train_weights'))
        netD_A_train_function.save(save_name.format('tf_D_A_train'))
        netD_A_train_function.save_weights(save_name.format('tf_D_A_train_weights'))
        netD_B_train_function.save(save_name.format('tf_D_B_train'))
        netD_B_train_function.save_weights(save_name.format('tf_D_B_train_weights'))
