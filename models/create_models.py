from keras import optimizers
from keras.models import Model
from keras.layers import Lambda
from models.loss import netG_loss, netD_real_loss, netD_fake_loss


def create_models(G_A_input, G_B_input, tensor_list, netD_A_input, netD_A_output, netD_B_input, netD_B_output):
    optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0)

    netG_train_function = Model([G_A_input, G_B_input], Lambda(netG_loss)(tensor_list))
    netG_train_function.compile('adam', 'mae')

    netDA_train_function_real_input = Model(netD_A_input, Lambda(netD_real_loss)(netD_A_output))
    netDA_train_function_real_input.compile('adam', 'mae')

    netDA_train_function_fake_input = Model(netD_A_input, Lambda(netD_fake_loss)(netD_A_output))
    netDA_train_function_fake_input.compile('adam', 'mae')

    netDB_train_function_real_input = Model(netD_B_input, Lambda(netD_real_loss)(netD_B_output))
    netDB_train_function_real_input.compile('adam', 'mae')

    netDB_train_function_fake_input = Model(netD_B_input, Lambda(netD_fake_loss)(netD_B_output))
    netDB_train_function_fake_input.compile('adam', 'mae')
    return netG_train_function, netDA_train_function_real_input, netDA_train_function_fake_input, netDB_train_function_fake_input, netDB_train_function_fake_input
