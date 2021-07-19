import tensorflow as tf
import tensorflow.keras as kr
import os

from tensorflow.python.keras.callbacks import CSVLogger


def training(model_name, dataset, checkpoint_name, csv_pathname, loss_func,
batch_size, num_epochs, last_epoch, optimizer, dont_load_weights = False, Show_summary = False):
  """
    Essa função recebe diversos parâmetros necessarios para o treino de uma rede neural e a treina usando o keras
  """

  #Base de dados

  x_train = dataset.x_train
  x_test = dataset.x_test
  y_train = dataset.y_train
  y_test = dataset.y_test

  #load do modelo 

  try:
    jsonFile = open("nNet_models/" + model_name, "r")
  except OSError:
    print("Fail atempt to load the model " + model_name)

  json_LDD_model = jsonFile.read()

  jsonFile.close()

  nNet = kr.models.model_from_json(json_LDD_model)

  #gerenciamento de pesos

  checkpoint_path = "checkpoints/" + checkpoint_name + "/" + checkpoint_name

  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
  
  
  directories = csv_pathname.split('/')
  if os.path.isdir(directories[0] +'/'+ directories[1] +'/'+ directories[2]):
    csv_callback = CSVLogger(filename = csv_pathname, append=True)
  else:
    os.mkdir(directories[0] +'/'+ directories[1] +'/'+ directories[2])
    csv_callback = CSVLogger(filename = csv_pathname, append=True)

  if(dont_load_weights == False):
    try:
      nNet.load_weights(checkpoint_path)
    except:
      print("Fail atempt to load wheights at " + checkpoint_path)

  if (Show_summary == True):
    nNet.summary()

  nNet.compile(optimizer = optimizer, loss = loss_func)

  history = nNet.fit(x=x_train, y=y_train, use_multiprocessing=True, initial_epoch = last_epoch + 1,
              callbacks=[cp_callback, csv_callback], validation_data=(x_test, y_test), batch_size = batch_size, epochs=last_epoch + 1 + num_epochs)


  return history