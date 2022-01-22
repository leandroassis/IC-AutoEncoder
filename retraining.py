from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError, BinaryCrossentropy
from modules.misc import LSSIM, ssim_metric, psnrb_metric, LossLinearCombination, AdversarialLoss
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard
from modules.TrainingFunctions import generator_training

environ["CUDA_VISIBLE_DEVICES"]="1"


for k in range(5):
    discriminator = KerasTrainingManager(training_idx = 0)
    generator = KerasTrainingManager(training_idx = 1)

    discriminator.change_parameters(dataset=DataSet().load_discriminator_training_set(training_idx = generator.training_idx))
    discriminator.start_training(epochs=2)
    generator.start_training(epochs=2)

