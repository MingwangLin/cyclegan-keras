import keras.backend as K
from keras.optimizers import Adam
from .networks import resnet_generator, n_layer_discriminator


def get_loss_fn(use_lsgan=True):
    if use_lsgan:
        loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
    else:
        loss_fn = lambda output, target: -K.mean(
            K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))
    return loss_fn


def get_model_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    compute_model_output = K.function([real_input, K.learning_phase()], [fake_output, rec_input])
    return real_input, fake_output, rec_input, compute_model_output


def get_loss_values(netD, real, fake, rec):
    output_real = netD([real])
    output_fake = netD([fake])

    loss_fn = get_loss_fn()
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))

    loss_D = loss_D_real + loss_D_fake
    loss_cyc = K.mean(K.abs(rec - real))
    return loss_D, loss_G, loss_cyc


def get_training_process(netGA, netGB, netDA, netDB, use_lsgan=True, lr=2e-4, beta1=0.5):
    if use_lsgan:
        lambda_param = 10
    else:
        lambda_param = 100

    real_A, fake_B, rec_A, _ = get_model_variables(netGA, netGB)
    real_B, fake_A, rec_B, _ = get_model_variables(netGB, netGA)

    loss_DA, loss_GB, loss_cycB = get_loss_values(netDA, real_A, fake_A, rec_A)
    loss_DB, loss_GA, loss_cycA = get_loss_values(netDB, real_B, fake_B, rec_B)
    loss_cyc = loss_cycB + loss_cycA

    loss_G = loss_GB + loss_GA + lambda_param * loss_cyc
    loss_D = loss_DA + loss_DB

    weightsD = netDA.trainable_weights + netDB.trainable_weights
    weightsG = netGB.trainable_weights + netGA.trainable_weights

    training_updates = Adam(lr=lr, beta_1=beta1).get_updates(weightsD, [], loss_D)
    netD_train = K.function([real_A, real_B, K.learning_phase()], [loss_DA / 2, loss_DB / 2], training_updates)

    training_updates = Adam(lr=lr, beta_1=beta1).get_updates(weightsG, [], loss_G)
    netG_train = K.function([real_A, real_B, K.learning_phase()], [loss_GB, loss_GA, loss_cyc], training_updates)
    return netD_train, netG_train
