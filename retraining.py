from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError, BinaryCrossentropy
from modules.misc import LSSIM, ssim_metric, psnrb_metric, AdversarialLoss
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard
from modules.TrainingFunctions import generator_training

environ["CUDA_VISIBLE_DEVICES"]="2"

def generator_discriminator_training():
    for k in range(1):
        discriminator = KerasTrainingManager(training_idx = 2)
        generator = KerasTrainingManager(training_idx = 3)

        discriminator.change_parameters(dataset=DataSet().load_discriminator_training_set(training_idx = generator.training_idx))
        discriminator.start_training(epochs=2)
        generator.change_parameters(loss = [MeanAbsoluteError, AdversarialLoss], loss_kwargs =  [None,{'training_idx' : 2}])
        generator.start_training(epochs=2)

def normal_training():
    pass

if __name__ == "__main__":
    generator_discriminator_training()
