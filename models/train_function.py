from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Lambda


def get_train_function(inputs, loss_function, lambda_layer_inputs):
    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0)
    train_function = Model(inputs, Lambda(loss_function)(lambda_layer_inputs))
    train_function.compile('adam', 'mae')
    return train_function
