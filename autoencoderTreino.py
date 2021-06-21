import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import os
from tensorflow._api.v2.image import ssim
from PIL import Image
from io import BytesIO
from scipy import misc

from DataMod import NoiseGen

def Ssim (i1, i2):
    return -ssim(i1, i2, max_val=255, filter_size = 11)

def main():

  #Base de dados

  (xTrain, yTrain),(xTest, yTest) = tf.keras.datasets.cifar10.load_data()

  #cifar em escala de cinza

  newXtrain = tf.image.rgb_to_grayscale(xTrain)
  newXtrain = np.array(newXtrain)

  newXtest = tf.image.rgb_to_grayscale(xTest)
  newXtest = np.array(newXtest)

  #adição compressão JPEG 
  newBase = []


  for indice in range(newXtrain.shape[0]):
    buffer = BytesIO()
    img = Image.fromarray(newXtrain[indice].reshape(32,32), mode="L")
    img.save(buffer, "JPEG", quality=30)
    image = Image.open(buffer)
    image = np.asarray(image)
    newBase.append(image)
    buffer.close()

  newTestBase = []

  for indice in range(newXtest.shape[0]):
    buffer = BytesIO()
    img = Image.fromarray(newXtest[indice].reshape(32,32), mode="L")
    img.save(buffer, "JPEG", quality=30)
    image = Image.open(buffer)
    image = np.asarray(image)
    newTestBase.append(image)
    buffer.close()



  #load do modelo 

  jsonFile = open("Unet.json", "r")

  json_LDD_model = jsonFile.read()

  jsonFile.close()

  nNet = kr.models.model_from_json(json_LDD_model)

  #gerenciamento de pesos

  checkpoint_path = "checkpoints6/cp.ckpt"

  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  
  #nNet.load_weights(checkpoint_path)


  #reshape após a compressão JPEG

  newBase = np.array(newBase)
  newTestBase = np.array(newTestBase)
  newBase = newBase.reshape(50000,32,32,1)
  newTestBase = newTestBase.reshape(10000,32,32,1)
  newBase = newBase.astype('float32')
  newTestBase = newTestBase.astype('float32')

  noiseDataTest = NoiseGen()
  noiseDataTest.grayLowNoiseMkr(newTestBase, 10)

  noiseData = NoiseGen()
  noiseData.grayLowNoiseMkr(newBase, 10)

  #nNet.summary()

  nNet.compile(optimizer=kr.optimizers.Adam(learning_rate=0.001), loss=Ssim)

  nNet.fit(x=noiseData.dataSet.astype('float32'), y=newXtrain.astype('float32'), 
           callbacks=[cp_callback], validation_data=(noiseDataTest.dataSet.astype('float32'), newXtest.astype('float32')), batch_size = 10, epochs=5)

main()