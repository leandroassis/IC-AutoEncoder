from unittest.mock import NonCallableMagicMock, NonCallableMock
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy, MeanSquaredError
from modules.CustomLosses import LSSIM, AdversarialLoss, L1AdversarialLoss
from modules.misc import psnrb_metric, ssim_metric, get_model
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard
from glob import glob

from modules.TrainingFunctions import *

environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np


### Data set

size = 2000

normal_distribution = tf.random.normal([size,100,1], 15, 2)
uniform_distribution = tf.random.uniform([size,100,1], -5, 5)

### Training

gan:Model = get_model(model_json_name = 'Generator-Dense-1.0.json')
discriminator:Model = get_model(model_json_name = "Discriminator-Dense-1.0.json")


discriminator.compile(optimizer = RMSprop(learning_rate=0.00001), loss = BinaryCrossentropy(), metrics = ['accuracy'])
gan.compile(optimizer = RMSprop(learning_rate=0.00001), loss = AdversarialLoss(model = discriminator))


for step in range(50):

    if step == 0:
        gan_predict = gan.predict(x = uniform_distribution)
        np.save("Init_gan_predict.npy", gan_predict)
    
    gan.fit(x= uniform_distribution, y = uniform_distribution, batch_size=10, epochs = 2)
    gan_predict = gan.predict(x = uniform_distribution)

    
    discriminator_x = tf.concat([gan_predict, tf.random.normal([size,100], 15, 2)], axis = 0)
    discriminator_y = tf.concat([tf.zeros([size,1]), tf.ones([size,1])], axis = 0)

    discriminator.fit(x= discriminator_x, y = discriminator_y, batch_size=10, epochs = 2)


gan_predict = gan.predict(x = uniform_distribution)

np.save("final_gan_predict.npy", gan_predict)
np.save("normal_distribution.npy", normal_distribution)