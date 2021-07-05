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
from DataMod import DataMod

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def Ssim (i1, i2):
    return -ssim(i1, i2, max_val=255, filter_size = 11)


#gerando a base
(xTrain, yTrain),(xTest, yTest) = tf.keras.datasets.cifar10.load_data()

test = DataMod(xTest)
test.rbg_to_gray()

test_target = test.dataSet.copy()
test.add_jpeg_compression_to_grayscale(compress_quality=30)
test.add_standard_Noise(max_pixel_var=10)

#Load do modelo
json_file = open('Unet.json', 'r')

loaded_json_file = json_file.read()

json_file.close()

nNet = kr.models.model_from_json(loaded_json_file)

checkpoint_path = "checkpoints6/cp.ckpt"

nNet.load_weights(checkpoint_path).expect_partial()


# predict
imgs1 = nNet.predict(test.dataSet)

imgs1 = (np.clip(imgs1, 0 , 255)).astype('uint8')



plt.figure(figsize=(20,4))

gaussImgs = gaussian_filter(test.dataSet, 0.38)

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(test_target[2*i+1050].reshape(32,32))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.savefig("img0.png")


plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(gaussImgs[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(gaussImgs[2*i+1050], test_target[2*i+1050].astype('uint8')).numpy(),2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.savefig("img1")


plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(test.dataSet[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(test.dataSet[2*i+1050], test_target[2*i+1050].astype('uint8')).numpy(),2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.savefig("img2")

plt.figure(figsize=(20,4))

for i in range (10):
  ax = plt.subplot(2,10, i+1)
  plt.imshow(imgs1[2*i+1050].reshape(32,32))
  plt.xlabel(round(-1*Ssim(imgs1[2*i+1050], test_target[2*i+1050].astype('uint8')).numpy(), 2))
  plt.gray()
  ax.get_yaxis().set_visible(False)

plt.savefig("img3")

