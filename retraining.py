from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError, BinaryCrossentropy
from modules.misc import LSSIM, L1AdversarialLoss, ssim_metric, psnrb_metric, AdversarialLoss
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard

environ["CUDA_VISIBLE_DEVICES"]="2"

def generator_discriminator_training():
    discriminator = KerasTrainingManager(training_idx = 9)
    generator = KerasTrainingManager(training_idx = 10)
    
    for k in range(3):
        discriminator.change_parameters(dataset=DataSet().load_discriminator_training_set(training_idx = generator.training_idx))
        discriminator.start_training(epochs=2)
        generator.change_parameters(loss = L1AdversarialLoss, loss_kwargs =  {'w1': 2, 'w2': 1, 'training_idx':discriminator.training_idx})
        generator.start_training(epochs=3)

def normal_training():
    generator = KerasTrainingManager(training_idx=47)
    generator.start_training(epochs=10)

if __name__ == "__main__":
    normal_training()
