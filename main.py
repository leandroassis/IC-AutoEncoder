from unittest.mock import NonCallableMock
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError, BinaryCrossentropy
from modules.misc import LSSIM, ssim_metric, psnrb_metric, AdversarialLoss, L1_AdversarialLoss, LSSIM_AdversarialLoss
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard

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


manager1 = KerasTrainingManager(
    "AutoEncoder-1.0-64x64.json",
    optimizer = Adam,
    optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
    loss = MeanAbsoluteError,
    loss_kwargs = {},
    compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
    
    fit_kwargs = {'batch_size' : 20, 'epochs' : 5, 'verbose':1, 'validation_split':0, 'shuffle':True, 
    'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
    'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

    metrics = [ssim_metric],

    callbacks = None,

    training_function = generator_training,

    dataset = DataSet().load_rafael_cifar_10_noise_data(),

    best_selector_metrics = [min, max]
)

manager1.start_training()


