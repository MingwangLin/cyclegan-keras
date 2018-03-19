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


def netG_loss(G_tensors, loss_weight=10):
    netD_A_predict_fake, rec_A, G_A_input, netD_B_predict_fake, rec_B, G_B_input = G_tensors

    loss_G_B = criterion_GAN(netD_A_predict_fake, K.ones_like(netD_A_predict_fake))
    loss_cyc_A = criterion_cycle(rec_A, G_A_input)

    loss_G_A = criterion_GAN(netD_B_predict_fake, K.ones_like(netD_B_predict_fake))
    loss_cyc_B = criterion_cycle(rec_B, G_B_input)

    loss_G = loss_G_A + loss_G_B + loss_weight * (loss_cyc_A + loss_cyc_B)

    return loss_G


def netD_loss(netD_predict):
    netD_predict_real, netD_predict_fake = netD_predict

    netD_loss_real = criterion_GAN(netD_predict_real, K.ones_like(netD_predict_real))
    netD_loss_fake = criterion_GAN(netD_predict_fake, K.zeros_like(netD_predict_fake))

    loss_netD = (1 / 2) * (netD_loss_real + netD_loss_fake)
    return loss_netD
