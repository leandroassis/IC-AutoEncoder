from sys import path
from os import getcwd, environ

path.insert(0, getcwd()) # adding IC-AutoEncoder folder to path
path.insert(0, getcwd() + '/modules/') # adding modules folder to path
from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM
from modules.misc import ssim_metric, psnrb_metric
from tensorflow.keras.optimizers import Adam

import mlflow.tensorflow


environ["CUDA_VISIBLE_DEVICES"] = "1"


mlflow.tensorflow.autolog()

tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()
tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)

autoEncoder.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = LSSIM(), metrics = [ssim_metric, psrnb_metric])

autoEncoder.fit(
        x = tinyDataSet.x_train,
        y = tinyDataSet.y_train,
        batch_size = 20,
        epochs = 15,
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

autoEncoder.evaluate()

autoEncoder.save_weights("nNet_models/" + model_name + ".h5")