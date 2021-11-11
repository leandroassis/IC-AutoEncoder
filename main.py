from tensorflow.keras.optimizers import Adam, SGD
from modules.misc import LSSIM, ssim_metric, psnr_metric
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard

environ["CUDA_VISIBLE_DEVICES"]="1"

obj = KerasTrainingManager(
    "Unet2.3-64x64.json",
    optimizer = Adam,
    optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
    loss = LSSIM,
    loss_kwargs = {'max_val' : 255, 'filter_size' : 9, 'filter_sigma' : 1.5, 'k1' :0.01, 'k2' :0.03},
    compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
    
    fit_kwargs = {'batch_size' : 20, 'epochs' : 5, 'verbose':1, 'validation_split':0, 'shuffle':True, 
    'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
    'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

    metrics = [ssim_metric, psnr_metric],

    dataset = DataSet().load_rafael_cifar_10_noise_data(),
    new = True
)

obj.start_training()