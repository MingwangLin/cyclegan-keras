from keras.layers import Input, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Lambda
from models.loss import netG_loss, netD_loss


def get_train_function(inputs, loss_function, lambda_layer_inputs):
    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0)
    train_function = Model(inputs, Lambda(loss_function)(lambda_layer_inputs))
    train_function.compile('adam', 'mae')
    return train_function


# create generator train function
def netG_train_function_creator(netD_A, netD_B, netG_A, netG_B, real_A, real_B, fake_A, fake_B):
    netD_B_predict_fake = netD_B(fake_B)
    rec_A = netG_B(fake_B)
    netD_A_predict_fake = netD_A(fake_A)
    rec_B = netG_A(fake_A)
    lambda_layer_inputs = [netD_B_predict_fake, rec_A, real_A, netD_A_predict_fake, rec_B, real_B]
    for l in netG_A.layers:
        l.trainable = True
    for l in netG_B.layers:
        l.trainable = True
    for l in netD_A.layers:
        l.trainable = False
    for l in netD_B.layers:
        l.trainable = False
    netG_train_function = get_train_function(inputs=[real_A, real_B], loss_function=netG_loss,
                                             lambda_layer_inputs=lambda_layer_inputs)
    return netG_train_function


# create discriminator A train function
def netD_A_train_function(netD_A, netD_B, netG_A, netG_B, real_A, finesize, input_nc):
    netD_A_predict_real = netD_A(real_A)
    _fake_A = Input(shape=(finesize, finesize, input_nc))
    _netD_A_predict_fake = netD_A(_fake_A)
    for l in netG_A.layers:
        l.trainable = False
    for l in netG_B.layers:
        l.trainable = False
    for l in netD_A.layers:
        l.trainable = True
    for l in netD_B.layers:
        l.trainable = False
    netD_A_train_function = get_train_function(inputs=[real_A, _fake_A], loss_function=netD_loss,
                                               lambda_layer_inputs=[netD_A_predict_real, _netD_A_predict_fake])
    return netD_A_train_function


# create discriminator B train function
def netD_A_train_function(netD_A, netD_B, netG_A, netG_B, real_B, finesize, input_nc):
    netD_B_predict_real = netD_B(real_B)
    _fake_B = Input(shape=(finesize, finesize, input_nc))
    _netD_B_predict_fake = netD_B(_fake_B)
    for l in netG_A.layers:
        l.trainable = False
        if isinstance(l, BatchNormalization):
            l._per_input_updates = {}
    for l in netG_B.layers:
        l.trainable = False
        if isinstance(l, BatchNormalization):
            l._per_input_updates = {}
    for l in netD_B.layers:
        l.trainable = True
    for l in netD_A.layers:
        l.trainable = False
    netD_B_train_function = get_train_function(inputs=[real_B, _fake_B], loss_function=netD_loss,
                                               lambda_layer_inputs=[netD_B_predict_real, _netD_B_predict_fake])
    return netD_B_train_function
