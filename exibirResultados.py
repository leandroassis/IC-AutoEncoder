import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from tensorflow.image import ssim
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from DataMod import NoiseGen



import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def Ssim (i1, i2):
    return -ssim(i1, i2, max_val=255, filter_size = 7)


(xTrain, yTrain),(xTest, yTest) = tf.keras.datasets.cifar10.load_data()

newXtest = tf.image.rgb_to_grayscale(xTest)
newXtest = np.array(newXtest)

noiseDataTest = NoiseGen()
noiseDataTest.grayLowNoiseMkr(newXtest, 19)


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
noiseDataTest.grayLowNoiseMkr(newTestBase, 19)


json_file = open('model5.json', 'r')

loaded_json_file = json_file.read()

json_file.close()

nNet = kr.models.model_from_json(loaded_json_file)

checkpoint_path = "checkpoints5/cp.ckpt"

nNet.load_weights(checkpoint_path)

imgs1 = nNet.predict(newTestBase)

imgs1 = imgs1.astype('uint8')

plt.figure(figsize=(20,4))


gaussImgs = gaussian_filter(noiseDataTest.dataSet, 0.45)

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
  plt.xlabel(-1*Ssim(gaussImgs[2*i+1050], newXtest[2*i+1050]).numpy())
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()


plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(noiseDataTest.dataSet[2*i+1050].reshape(32,32))
  plt.xlabel(-1*Ssim(noiseDataTest.dataSet[2*i+1050], newXtest[2*i+1050]).numpy())
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()

plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(imgs1[2*i+1050].reshape(32,32))
  plt.xlabel(-1*Ssim(imgs1[2*i+1050], newXtest[2*i+1050]).numpy())
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.show()

