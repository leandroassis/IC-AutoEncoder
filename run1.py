from sys import path
from os import getcwd, environ

from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM
from modules.misc import ssim_metric, psnrb_metric
from tensorflow.keras.optimizers import Adam

import mlflow.keras

from models.autoEncoder import autoEncoder

environ["CUDA_VISIBLE_DEVICES"] = "3"


tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()
tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)

autoEncoder.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = LSSIM(), metrics = [ssim_metric, psnrb_metric])

mlflow.keras.autolog()

history = autoEncoder.fit(
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

score = autoEncoder.evaluate(tinyDataSet.x_test, tinyDataSet.y_test, batch_size = 20, verbose = 1)

autoEncoder.save_weights("models/weights" + autoEncoder.model_name + ".h5")