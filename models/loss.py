import keras.backend as K


def criterion_GAN(output, target, use_lsgan=True):
    if use_lsgan:
        diff = output - target
        dims = list(range(1, K.ndim(diff)))
        return K.expand_dims((K.mean(diff ** 2, dims)), 0)
    else:
        return K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))


def criterion_cycle(rec, real):
    diff = K.abs(rec - real)
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims((K.mean(diff, dims)), 0)


def get_model_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    compute_model_output = K.function([real_input, K.learning_phase()], [fake_output, rec_input])
    return real_input, fake_output, rec_input, compute_model_output


def netG_loss(G_tensors, loss_weight=10):

    netD_A_pred_fake, rec_A, G_A_input, netD_B_pred_fake, rec_B, G_B_input = G_tensors

    loss_G_B = criterion_GAN(netD_A_pred_fake, K.ones_like(netD_A_pred_fake))
    loss_cyc_A = criterion_cycle(rec_A, G_A_input)

    loss_G_A = criterion_GAN(netD_B_pred_fake, K.ones_like(netD_B_pred_fake))
    loss_cyc_B = criterion_cycle(rec_B, G_B_input)

    loss_G = loss_G_A + loss_G_B + loss_weight * (loss_cyc_A + loss_cyc_B)

    return loss_G


def netD_real_loss(D_pred):

    loss_D_real = criterion_GAN(D_pred, K.ones_like(D_pred))
    loss_D_real = (1 / 2) * loss_D_real

    return loss_D_real


def netD_fake_loss(D_pred):

    loss_D_fake = criterion_GAN(D_pred, K.zeros_like(D_pred))
    loss_D_fake = (1 / 2) * loss_D_fake

    return loss_D_fake
