from unittest.mock import NonCallableMagicMock, NonCallableMock
from tensorflow.keras.optimizers import Adam, SGD
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

environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np


### Data set

size = 500

normal_distribution = tf.random.normal([size,1], 2, 2)
uniform_distribution = tf.random.uniform([size,1], -5, 5)

### Training

gan:Model = get_model(model_json_name = 'Generator-Dense1-2.json')
discriminator:Model = get_model(model_json_name = "Discriminator-Dense1-2.json")


discriminator.compile(optimizer = Adam(learning_rate=0.0001), loss = BinaryCrossentropy(), metrics = ['accuracy'])
gan.compile(optimizer = Adam(learning_rate=0.0001), loss = AdversarialLoss(model = discriminator))

discriminator_x = np.array(uniform_distribution)
discriminator_y = tf.ones([size,1])

for step in range(20):

    gan.fit(x= uniform_distribution, y = uniform_distribution, batch_size=10, epochs = 7)
    gan_predict = gan.predict(x = uniform_distribution)

    if step == 0:
        np.save("Init_gan_predict.npy", gan_predict)

    discriminator_x = tf.concat([discriminator_x, gan_predict, tf.random.normal([size,1], 2, 2)], axis = 0)
    discriminator_y = tf.concat([discriminator_y, tf.zeros([size,1]), tf.ones([size,1])], axis = 0)

    discriminator.fit(x= discriminator_x, y = discriminator_y, batch_size=10, epochs = 2)


gan_predict = gan.predict(x = uniform_distribution)

np.save("final_gan_predict.npy", gan_predict)
np.save("normal_distribution.npy", normal_distribution)