import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import os
from tensorflow._api.v2.image import ssim
from scipy import misc

from DataMod import DataMod

def Ssim (i1, i2):
    return -ssim(i1, i2, max_val=255, filter_size = 11)

def main():

  #Base de dados

  (xTrain, yTrain),(xTest, yTest) = tf.keras.datasets.cifar10.load_data()

  train = DataMod(xTrain)
  test = DataMod(xTest)

  # Alvo

  train_target = DataMod(xTrain)
  train_target.rbg_to_gray()

  test_target = DataMod(xTest)
  test_target.rbg_to_gray()

  #cifar em escala de cinza

  train.rbg_to_gray()
  test.rbg_to_gray()

  #adicionando a compressão JPEG

  train.add_jpeg_compression_to_grayscale(compress_quality=5)
  test.add_jpeg_compression_to_grayscale(compress_quality=5)

  #adicionando ruído

  train.add_standard_Noise(max_pixel_var=10)
  test.add_standard_Noise(max_pixel_var=10)

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

  #nNet.summary()

  nNet.compile(optimizer=kr.optimizers.Adam(learning_rate=0.001), loss=Ssim)

  nNet.fit(x=train.dataSet.astype('float32'), y=train_target.dataSet.astype('float32'), 
           callbacks=[cp_callback], validation_data=(test.dataSet.astype('float32'), test_target.dataSet.astype('float32')), batch_size = 10, epochs=2)

main()