from sys import path
from os import getcwd, environ

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3,1"

from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM, LPSNRB, L3SSIM
from modules.misc import ssim_metric
from modules.ImageMetrics.metrics import three_ssim, psnrb
from tensorflow.keras.optimizers import Adam

from keras import models

import mlflow.keras

import multiprocessing


# ## Fetching Datasets

# creates the datasets
tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()

tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()

# concatenates the datasets
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)


# ## Training Models

# fix bath_size and epochs (how to decide the number of epochs and batch size?)
batch_size = 20
epochs = 15

mlflow.keras.autolog()

# trains a model with a datasets
def train_models(dataset : DataSet):

        file = open("logs/logs.txt", "w")

        # training for each loss
        losses = {"LSSIM":LSSIM(), "LPSNRB":LPSNRB(), "L3SSIM":L3SSIM()}

        # for each loss
        for loss in losses:
                # train each model
                for path in ["models/arch/AutoEncoder-2.3-64x64.json", "models/arch/ResidualAutoEncoder-0.1-64x64.json", "models/arch/Unet2.3-64x64.json"]:
                        # reads the model
                        with open(path, "r") as json_file:
                                model = models.model_from_json(json_file.read())
                        
                        try:
                                model.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = losses[loss], metrics = [ssim_metric, three_ssim, psnrb])
                        except Exception as e:
                                file.write(f"Error {e}: Error compiling {model.name} with {dataset.name} dataset and loss {loss}\n")
                                continue
                        

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
                                                class_weNoneight = None,
                                                sample_weight = None,
                                                steps_per_epoch = None,
                                                validation_steps = None,
                                                validation_batch_size = None,
                                                validation_freq = 1,
                                                max_queue_size = 10,
                                                workers = 1,
                                                use_multiprocessing = False
                                        )

                                        model.save_weights("logs/run1/weights/" + model.name + dataset.name + loss +".h5")

                                except Exception as e:
                                        file.write(f"Error {e}: Error fitting and saving {model.name} with {dataset.name} dataset and loss {loss}\n")
        file.close()


procs = []

# to do: paralelize the training
'''
multiprocessing.set_start_method('spawn')

for dataset in [tinyDataSet, cifarDataSet, cifarAndTinyDataSet]:
        proc = multiprocessing.Process(target=train_model, args=(dataset, ))
        procs.append(proc)

# waits for the training to finish
for proc in procs:
        proc.start()
        proc.join()
'''

# for each dataset
for dataset in [tinyDataSet, cifarDataSet, cifarAndTinyDataSet]:
        train_models(dataset)