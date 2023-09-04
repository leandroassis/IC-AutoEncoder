import keras_tuner as kt

from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
path.insert(0, getcwd() + "/models/")

environ["CUDA_VISIBLE_DEVICES"] = "1"

from models.functions.autoEncoder import create_AE_model
from modules.DataMod import DataSet

from tensorflow.math import exp
from tensorflow.keras.callbacks import LearningRateScheduler


def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001
    else:
        return lr * 0.85

print("Fetching datasets...")
cifar_tiny = DataSet(), DataSet(), DataSet()
cifar_tiny = cifar_tiny.load_cifar_and_tiny()
print("Datasets fetched!")

print("Adding gaussian noise...")
cifar_tiny = cifar_tiny.add_gaussian_noise(0.3)
print("Gaussian noise added!")

tuner = kt.BayesianOptimization(create_AE_model,
                  objective= kt.Objective('val_three_ssim', direction="max"),
                  max_trials=60,
                  max_retries_per_trial=3,
                  max_consecutive_failed_trials=5,
                  beta=3.0)

tuner.search_space_summary()
tuner.search(cifar_tiny.x_train, cifar_tiny.y_train, validation_data=(cifar_tiny.x_test, cifar_tiny.y_test), callbacks = [ LearningRateScheduler(scheduler) ])

hps = tuner.get_best_hyperparameters(25)

with open("models/models_params_counter.csv", "w") as file:
    file.write('model_name, num_params, ssim, tssim, psnrb,')
    
    for key, value in hps[0].values.items():
        file.write(str(key)+',')
        
    file.write('\n')


for hp in hps:
    model = tuner.hypermodel.build(hp)

    model.fit(cifar_tiny.x_train, cifar_tiny.y_train, epochs=15, batch_size=20 , callbacks = [ LearningRateScheduler(scheduler) ])

    loss, ssim, tssim, psnrb = model.evaluate(cifar_tiny.x_test, cifar_tiny.y_test)

    with open("models/models_params_counter.csv", "a") as file:
        file.write(str(model.model_name)+','+str(model.count_params())+','+','+str(ssim)+','+str(tssim)+','+str(psnrb)+',')

        for key, value in hp.values.items():
            file.write(str(value)+',')
        
        file.write('\n')

