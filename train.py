import time
import numpy as np
import keras.backend as K
from IPython.display import clear_output
from options.train_options import TrainOptions
from data.data_loader import load_data, minibatchAB
from data.data_display import show_generator_image
from models.train_function import get_train_function
from models.networks import resnet_generator, n_layer_discriminator
from models.loss import netG_loss, netD_loss

opt = TrainOptions().parse()

# load data
dpath = opt.dataroot

train_A = load_data(dpath + 'trainA/*')
train_B = load_data(dpath + 'trainB/*')
print('#training images = {}'.format(len(train_A)))
train_batch = minibatchAB(train_A, train_B, batch_size=opt.batch_size)

val_A = load_data(dpath + 'valA/*')
val_B = load_data(dpath + 'valB/*')
print('#training images = {}'.format(len(train_A)))
val_batch = minibatchAB(val_A, val_B, batch_size=4)

# create gennerator models
netG_A, real_A, fake_B = resnet_generator()
netG_B, real_B, fake_A = resnet_generator()
netG_A.summary()

# create discriminator models
netD_A = n_layer_discriminator()
netD_B = n_layer_discriminator()
netD_A.summary()

# create generator train function
netD_B_predict_fake = netD_B(fake_B)
rec_A = netG_B(fake_B)
netD_A_predict_fake = netD_A(fake_A)
rec_B = netG_A(fake_A)
lambda_layer_inputs = [netD_B_predict_fake, rec_A, real_A, netD_A_predict_fake, rec_B, real_B]

netG_train_function = get_train_function(inputs=[real_A, real_B], loss_function=netG_loss,
                                         lambda_layer_inputs=lambda_layer_inputs)

# create discriminator A train function
netD_A_predict_real = netD_A(real_A)

for l in netG_A.layers:
    l.trainable = False
for l in netG_B.layers:
    l.trainable = False

netD_A_train_function = get_train_function(inputs=[real_A, real_B], loss_function=netD_loss,
                                           lambda_layer_inputs=[netD_A_predict_real, netD_A_predict_fake])
# create discriminator B train function
netD_B_predict_real = netD_B(real_B)

for l in netG_A.layers:
     l.trainable=False
for l in netG_B.layers:
     l.trainable=False

netD_B_train_function = get_train_function(inputs=[real_B, real_B], loss_function=netD_loss,
                                           lambda_layer_inputs=[netD_B_predict_real, netD_B_predict_fake])

# train loop

time_start = time.time()
how_many_epochs = 5
iteration_count = 0
epoch_count = 0
batch_size = 1
display_freq = 100

while epoch_count < how_many_epochs:
    target_label = np.zeros((batch_size, 1))
    epoch_count, A, B = next(train_batch)

    netD_B_train_function.train_on_batch([A, B], target_label)
    netD_A_train_function.train_on_batch([A, B], target_label)

    netG_train_function.train_on_batch([A, B], target_label)

    iteration_count += 1

    if iteration_count % display_freq == 0:
        clear_output()
        traintime = (time.time() - time_start) / iteration_count
        print('epoch_count: {}/{}  iter_count: {}  timecost/iter: {}s'.format(epoch_count, niter, iteration_count,
                                                                              traintime))
        _, val_A, val_B = next(val_batch)
        show_generator_image(val_A, val_B, netG_A, netG_B)

        save_name = dpath + '{}' + str(iteration_count) + '.h5'

        netG_A.save(save_name.format('tf_GA'))
        netG_A.save_weights(save_name.format('tf_GA_weights'))
        netG_B.save(save_name.format('tf_GB'))
        netG_B.save_weights(save_name.format('tf_GB_weights'))
        netD_A.save(save_name.format('tf_DA'))
        netD_A.save_weights(save_name.format('tf_DA_weights'))
        netD_B.save(save_name.format('tf_DB'))
        netD_B.save_weights(save_name.format('tf_DB_weights'))

        netG_train_function.save(save_name.format('tf_G_train'))
        netG_train_function.save_weights(save_name.format('tf_G_train_weights'))
        netD_A_train_function.save(save_name.format('tf_D_A_train'))
        netD_A_train_function.save_weights(save_name.format('tf_D_A_train_weights'))
        netD_B_train_function.save(save_name.format('tf_D_B_train'))
        netD_B_train_function.save_weights(save_name.format('tf_D_B_train_weights'))
