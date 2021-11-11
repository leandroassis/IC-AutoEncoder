import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from misc import Ssim

import logging
import os

#desativando warnings do TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def nNet_result_data(image_name, dataset, checkpoint_name,  model_name, sigma):
  '''
   Salva uma imagem com resultados do treino da Rede Neural, e retorna média junto do desvio padrão \n
   para os resultados
  '''
  x_test = dataset.x_test
  y_test = dataset.y_test

  #Load do modelo
  json_file = open('nNet_models/' + model_name, 'r')

  loaded_json_file = json_file.read()

  json_file.close()

  nNet = kr.models.model_from_json(loaded_json_file)

  checkpoint_path = "checkpoints/" + checkpoint_name + "/" +checkpoint_name

  nNet.load_weights(checkpoint_path).expect_partial()

  # predict
  nNet_imgs = nNet.predict(x_test)

  nNet_imgs = (np.clip(nNet_imgs, 0 , 255)).astype('uint8')

  # Uso do filtro gaussiano
  gaussImgs = gaussian_filter(x_test, sigma=(0, sigma, sigma, 0))

  # gerando as imagens de comparação
  fig = plt.figure(figsize=(16,9))

  for i in range (10):
    ax = fig.add_subplot(4, 10, i+1)
    plt.imshow(y_test[431*i+1050].reshape(64, 64))
    if (i == 5):
      plt.title('Imagens Alvo')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


  for i in range (10):
    ax = fig.add_subplot(4, 10, i+11)
    plt.imshow(x_test[431*i+1050].reshape(64, 64))
    plt.xlabel(round(-1*Ssim(x_test[2*i+1050].astype('uint8'), y_test[2*i+1050].astype('uint8')).numpy(),3))
    if (i == 5):
      plt.title('Imagens com ruído (entrada da rede)')
    plt.gray()
    ax.get_yaxis().set_visible(False)



  for i in range (10):
    ax = fig.add_subplot(4, 10, i+21)
    plt.imshow(gaussImgs[431*i+1050].reshape(64, 64))
    # Coloca o resultado do ssim entre a imagem original e a
    plt.xlabel(round(-1*Ssim(gaussImgs[2*i+1050].astype('uint8'), y_test[2*i+1050].astype('uint8')).numpy(),3))
    if (i == 5):
      plt.title('Aplicação do filtro gaussiano nas imagens com ruído (sigma = ' + str(sigma) + ')')
    plt.gray()
    ax.get_yaxis().set_visible(False)



  for i in range (10):
    ax = fig.add_subplot(4, 10, i+31)
    plt.imshow(nNet_imgs[431*i+1050].reshape(64, 64))
    plt.xlabel(round(-1*Ssim(nNet_imgs[2*i+1050].astype('uint8'), y_test[2*i+1050].astype('uint8')).numpy(), 3))
    if (i == 5):
      plt.title('Resultado da rede neural')
    plt.gray()
    ax.get_yaxis().set_visible(False)


  if os.path.isdir('Relatorios-Dados-etc/Imagens de resultados/' + image_name.split('/')[0]):
    plt.savefig('Relatorios-Dados-etc/Imagens de resultados/' + image_name)
  else:
    os.mkdir('Relatorios-Dados-etc/Imagens de resultados/' + image_name.split('/')[0])
    plt.savefig('Relatorios-Dados-etc/Imagens de resultados/' + image_name)

  ssim_gauss = Ssim(gaussImgs.astype('uint8'), y_test.astype('uint8')).numpy()

  ssim_nNet = Ssim(nNet_imgs.astype('uint8'), y_test.astype('uint8')).numpy()

  ssim_base = Ssim(x_test.astype('uint8'), y_test.astype('uint8')).numpy()

  return (ssim_gauss.mean(), ssim_gauss.std()), (ssim_nNet.mean(), ssim_nNet.std()), (ssim_base.mean(), ssim_base.std())

