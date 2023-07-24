#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sys import path
from os import getcwd, environ

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3,1"

from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM
from modules.misc import ssim_metric
from modules.ImageMetrics.metrics import three_ssim
from tensorflow.keras.optimizers import Adam

import mlflow.keras

from models.autoEncoder import autoEncoder
from models.ResidualAutoencoder import residualAutoEncoder
from models.Unet import unet

import multiprocessing
from datetime import datetime


# ## Fetching Datasets

# In[ ]:


# creates the datasets
tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()

tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()

# concatenates the datasets
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)


# ## Training Models

# In[ ]:


# to do: 
# paralelize the training (does it's necessary?)
# batch size shoudn't be specified (keras API doc), does it affect the training?

# training with LSSIM loss function and ssim and psnrb metrics


# In[ ]:


# compiles all the models

autoEncoder.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = LSSIM(), metrics = [ssim_metric, three_ssim])
unet.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = LSSIM(), metrics = [ssim_metric, three_ssim])
residualAutoEncoder.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = LSSIM(), metrics = [ssim_metric, three_ssim])


# In[ ]:


batch_size = 20
epochs = 15
# how to decide the number of epochs and batch size?

file = open("logs/run1.txt", "w")

mlflow.keras.autolog()

# function to parallelize the training
def train_model_paralel(model):
        # trains the models with the datasets
        print(model.name)
        for dataset in [tinyDataSet, cifarDataSet, cifarAndTinyDataSet]:
                with mlflow.start_run(run_name= model.name + dataset.name):
                        
                        try:
                                history = model.fit(
                                        x = dataset.x_train,
                                        y = dataset.y_train,
                                        batch_size = batch_size,
                                        epochs = epochs,
                                        verbose = 1,
                                        validation_split = 0,
                                        shuffle = True,
                                        class_weight = None,
                                        sample_weight = None,
                                        steps_per_epoch = None,
                                        validation_steps = None,
                                        validation_batch_size = None,
                                        validation_freq = 1,
                                        max_queue_size = 10,
                                        workers = 1,
                                        use_multiprocessing = False
                                )
                                model.save_weights("models/weights/run1/" + model.name + dataset.name + ".h5")

                                score = model.evaluate(dataset.x_test, dataset.y_test, verbose = 1)

                        except Exception as e:
                                file.write(f"Error {e}: Error training {model.name} with {dataset.name} dataset\n")
                                file.write(e.__cause__)
                                file.write(e.__context__)


# In[ ]:


start_date = datetime.now()

procs = []

#multiprocessing.set_start_method("spawn")

# launches the training in parallel
for model in [autoEncoder, unet, residualAutoEncoder]:
        train_model_paralel(model)
        #proc = multiprocessing.Process(target=train_model_paralel, args=(model, ))
        #proc.start()
        #procs.append(proc)

# waits for the training to finish
#for proc in procs:
#        proc.join()

end_date = datetime.now()


# In[ ]:


file.write("Start date: " + str(start_date) + "\n")
file.write("End date: " + str(end_date) + "\n")
file.write("Duration: " + str(end_date - start_date) + "\n")

