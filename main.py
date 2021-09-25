from automatic_training import Auto_Training
from misc import LSSIM
from tensorflow.keras.optimizers import Adam, SGD
from os import environ

environ["CUDA_VISIBLE_DEVICES"]="1"

at = Auto_Training()
at.state.show()
'''
at.set_a_new_training(
    {
        'model_name': 'AutoEncoder-1.0-64x64.json',
        'number_of_epochs' : 15,
        'optimizer' : SGD,
        '*optimizer_kwargs' : {
            'learning_rate' : 0.1,
            'momentum' : 0.9,
            'nesterov' : False
        },
        'fit_Kwargs' : {
            'batch_size' : 20
        }
    }
)
'''