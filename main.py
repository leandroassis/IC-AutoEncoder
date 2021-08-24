from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import tensorflow
import os

from sys import version
from tensorflow import __version__ as tensorflow_version
from scipy import __version__ as scipy_version
from numpy import __version__ as numpy_version
from tensorflow.keras import __version__ as keras_version
from tensorflow_addons import __version__ as tensorflow_addons_version
python_version = version[:6]

from DataMod import DataSet

from tensorflow._api.v2.image import ssim
from tensorflow.keras.losses import mse
from tensorflow_addons.optimizers import CyclicalLearningRate as CLR

from training import training
from show_results import nNet_result_data
from finding_best_sigma import find_best_sigma_for_ssim
from misc import get_current_time_and_data , get_last_epoch, Ssim, write_params_log

date_str, time_str = get_current_time_and_data()

os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Dataset
training_idx = 11
dataset = DataSet()
dataset.load_rafael_tinyImagenet_64x64_noise_data()

# training param
model_name = "Unet2.0-64x64.json"
checkpoint_name = model_name.replace('.json', '-#' + str(training_idx)) + '-' + dataset.name
csv_name = checkpoint_name
csv_pathname = "Relatorios-Dados-etc/Parametros e dados de Resultados/" + model_name.replace('.json', '|') + dataset.name + "/" + csv_name + ".log"
learning_rate = 0.001
batch_size = 20
num_epochs = 3
last_epoch = get_last_epoch(csv_pathname)
actual_epoch = last_epoch + num_epochs


#compile param
loss_func = Ssim

# CLR params
use_CLR = False
initial_learning_rate = learning_rate
maximal_learning_rate=1e-2
step_size=2000
scale_fn = lambda x: 1.
scale_mode="cycle"
name="MyCyclicScheduler"
clr = CLR(
    initial_learning_rate = learning_rate, 
    maximal_learning_rate=maximal_learning_rate,
    step_size=step_size,
    scale_fn=scale_fn,
    scale_mode=scale_mode,
    name=name)

# optimizer

optimizer = Adam(learning_rate = clr if use_CLR else learning_rate)


#param de exibição (não citados antes)

image_name = checkpoint_name + "/" + date_str + "|" + time_str + "|" + checkpoint_name + "|" + "|epoch=" + str(actual_epoch) + ".png"

sigma = find_best_sigma_for_ssim(dataset.x_test, dataset.y_test)

history = training(model_name = model_name, dataset = dataset, checkpoint_name = checkpoint_name, csv_pathname = csv_pathname,
optimizer = optimizer, loss_func = loss_func, batch_size = batch_size, num_epochs = num_epochs, last_epoch = last_epoch)

(ssim_gauss_mean, ssim_gauss_std), (ssim_nNet_mean, ssim_nNet_std), (ssim_base_mean, ssim_base_std) = nNet_result_data(image_name = image_name,
dataset = dataset, checkpoint_name = checkpoint_name,  model_name = model_name, sigma = sigma)


# gerando o Json com os parametros de treino
Nnet_params_results = {

    "versions" : 
    {
        "python_version" : python_version,
        "tensorflow_version" : tensorflow_version ,
        "tensorflow_addons_version" : tensorflow_addons_version,
        "scipy_version" : scipy_version,
        "numpy_version" : numpy_version,
        "keras_version" : keras_version
    },

    "model_name" : model_name,
    "dataset_name" : dataset.name,
    "training_idx" : training_idx,
    "bath_size" : batch_size,
    "last_epoch" : actual_epoch,

    "optimizer_params" : {
        "name" : optimizer._name,
        "init_learning_rate" : learning_rate,
        "momentun" : 0,
        "nesterov" : False,
        "Ciclical_learning_rate" :{
            "used" : use_CLR,
            "initial_learning_rate" : initial_learning_rate,
            "maximal_learning_rate" : maximal_learning_rate,
            "step_size" : step_size,
            "scale_fn" : 1.0,
            "scale_mode" : scale_mode,
            "name" : name
        }
    },
    
    "loss_func" : {
        "name" : loss_func.__name__,
    },

    "last_results" : 
    {
        'nNet_mean_ssim' :  float(ssim_nNet_mean),
        'nNet_std_ssim' :  float(ssim_nNet_std),
        'gauss_mean_ssim' : float(ssim_gauss_mean),
        'gauss_std_ssim' : float(ssim_gauss_std),
        'base_mean_ssim' : float(ssim_base_mean),
        'base_std_ssim' : float(ssim_base_std)
    }

}

json_log_name = checkpoint_name + ".json"
json_log_file_pathname = "Relatorios-Dados-etc/Parametros e dados de Resultados/" + model_name.replace('.json', '|') + dataset.name + "/" + json_log_name
write_params_log(json_log_file_pathname, Nnet_params_results)