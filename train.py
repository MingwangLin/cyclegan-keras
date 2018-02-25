import time
from options.train_options import TrainOptions
from data.data_loader import load_data, minibatchAB
from models.loss import get_training_process
from models.networks import resnet_generator, n_layer_discriminator

opt = TrainOptions().parse()

train_A = load_data(opt.dataroot + 'trainA/*')
train_B = load_data(opt.dataroot + 'trainB/*')
print('#training images = {}'.format(len(train_A)))
train_batch = minibatchAB(train_A, train_B, opt.batchSize)

netGB = resnet_generator()
netGA = resnet_generator()
netGA.summary()

netDA = n_layer_discriminator()
netDB = n_layer_discriminator()
netDA.summary()

netD_train, netG_train = get_training_process(netGA, netGB, netDA, netDB)

total_steps = 0
epoch_count = 0
errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
save_latest_freq = opt.save_latest_freq

while epoch_count < opt.niter:
    time_start = time.time()

    epoch_count, A, B = next(train_batch)

    errDA, errDB = netD_train([A, B, 1])
    errDA_sum += errDA
    errDB_sum += errDB

    errGA, errGB, errCyc = netG_train([A, B, 1])
    errGA_sum += errGA
    errGB_sum += errGB
    errCyc_sum += errCyc
    total_steps += 1
    if total_steps % save_latest_freq == 0:
        print('[%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc %f'
              % (epoch_count, opt.niter, total_steps, errDA_sum / save_latest_freq, errDB_sum / save_latest_freq,
                 errGA_sum / save_latest_freq, errGB_sum / save_latest_freq,
                 errCyc_sum / save_latest_freq), time.time() - time_start)

        save_name = opt.dataroot + '{}' + str(total_steps) + '{}'

        netGA.save(save_name.format('tf_GA', '.h5'))
        netGA.save_weights(save_name.format('tf_GA_weights', '.h5'))

        netGB.save(save_name.format('tf_GB', '.h5'))
        netGB.save_weights(save_name.format('tf_GB_weights', '.h5'))

        netDA.save(save_name.format('tf_DA', '.h5'))
        netDA.save_weights(save_name.format('tf_DA_weights', '.h5'))

        netDB.save(save_name.format('tf_DB', '.h5'))
        netDB.save_weights(save_name.format('tf_DB_weights', '.h5'))

        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
