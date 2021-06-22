import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from tensorflow._api.v2.image import ssim
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from DataMod import NoiseGen

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def Ssim (i1, i2):
    return -ssim(i1, i2, max_val=255, filter_size = 11)


(xTrain, yTrain),(xTest, yTest) = tf.keras.datasets.cifar10.load_data()

newXtest = tf.image.rgb_to_grayscale(xTest)
newXtest = np.array(newXtest)

del xTrain, xTest, yTrain, yTest

newTestBase = []

for indice in range(newXtest.shape[0]):
  buffer = BytesIO()
  img = Image.fromarray(newXtest[indice].reshape(32,32), mode="L")
  img.save(buffer, "JPEG", quality=30)
  image = Image.open(buffer)
  image = np.asarray(image)
  newTestBase.append(image)
  buffer.close()

del buffer

newTestBase = np.array(newTestBase)
newTestBase = newTestBase.reshape(10000,32,32,1)

noiseDataTest = NoiseGen()
noiseDataTest.grayLowNoiseMkr(newTestBase, 10)

noiseDataTest.dataSet = noiseDataTest.dataSet

json_file = open('Unet.json', 'r')

loaded_json_file = json_file.read()

json_file.close()

nNet = kr.models.model_from_json(loaded_json_file)

checkpoint_path = "checkpoints6/cp.ckpt"

nNet.load_weights(checkpoint_path).expect_partial()

imgs1 = nNet.predict(noiseDataTest.dataSet)

imgs1 = (np.clip(imgs1, 0 , 255)).astype('uint8')



plt.figure(figsize=(20,4))

gaussImgs = gaussian_filter(noiseDataTest.dataSet, 0.38)

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(newXtest[2*i+1050].reshape(32,32))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()


plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(gaussImgs[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(gaussImgs[2*i+1050], newXtest[2*i+1050].astype('uint8')).numpy(),2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()


plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(noiseDataTest.dataSet[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(noiseDataTest.dataSet[2*i+1050], newXtest[2*i+1050].astype('uint8')).numpy(),2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()

plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(imgs1[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(imgs1[2*i+1050], newXtest[2*i+1050].astype('uint8')).numpy(), 2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()

