from unittest.mock import NonCallableMagicMock, NonCallableMock
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError, BinaryCrossentropy
from modules.misc import LSSIM, AdversarialLoss, L1AdversarialLoss, ssim_metric
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard
from glob import glob

from modules.TrainingFunctions import *

environ["CUDA_VISIBLE_DEVICES"]="3"

'''
manager2 = KerasTrainingManager(
    "Discriminator-AutoEncoder-1.0-64x64.json",
    optimizer = Adam,
    optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
    loss = BinaryCrossentropy,
    loss_kwargs = {},
    compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
    
    fit_kwargs = {'batch_size' : 20, 'epochs' : 1, 'verbose':1, 'validation_split':0, 'shuffle':True, 
    'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
    'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

    metrics = ['accuracy'],

    callbacks = None,
    
    training_function = generator_training,

    dataset = DataSet().load_discriminator_training_set(generator_name = "AutoEncoder-1.0-64x64.json"),
    best_selector_metrics = [min, max]
)

manager2.start_training()
'''
'''
models = glob(f"./nNet_models/*.json", recursive = True)
models.remove('./nNet_models/Discriminator-AutoEncoder-1.0-64x64.json')
models.reverse()
model = model[14:]
'''

for model in ['Conv-1.0-64x64.json']:

    #for optimizer, optimizer_kwargs in [(SGD, {'learning_rate':0.01, 'momentum':0.95} )]:

    manager1 = KerasTrainingManager(
        model,
        optimizer = Adam,
        optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
        loss = LSSIM,
        loss_kwargs = {'max_val':255, 'filter_size':9, 'filter_sigma':1.5, 'k1':0.01, 'k2':0.03},
        compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
        
        fit_kwargs = {'batch_size' : 20, 'epochs' : 20, 'verbose':1, 'validation_split':0, 'shuffle':True, 
        'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
        'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

        metrics = [ssim_metric],

        callbacks = None,

        training_function = generator_training,

        dataset = DataSet().load_rafael_cifar_10_noise_data(),

        best_selector_metrics = [min, max]
    )

    manager1.start_training()


