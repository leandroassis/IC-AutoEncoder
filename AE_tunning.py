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
        return 0.01
    else:
        return lr * 0.85


tuner = kt.BayesianOptimization(create_AE_model,
                  objective= [kt.Objective('val_three_ssim', direction="max"), kt.Objective('val_psnrb', direction="max"), kt.Objective('val_ssim_metric', direction="max")],
                  max_trials=60,
                  max_epochs=20,
                  executions_per_trial=1)

tuner.search_space_summary()

cifar, tiny, cifar_tiny = DataSet(), DataSet(), DataSet()

cifar = cifar.load_rafael_cifar_10_noise_data()
tiny = tiny.load_rafael_tinyImagenet_64x64_noise_data()

cifar_tiny = cifar_tiny.concatenateDataSets(cifar, tiny)
cifar_tiny = cifar_tiny.add_gaussian_noise(0.1)

tuner.search(cifar_tiny.x_train, cifar_tiny.y_train, validation_data=(cifar_tiny.x_test, cifar_tiny.y_test), callbacks = [ LearningRateScheduler(scheduler) ])

hps = tuner.get_best_hyperparameters(25)

with open("models/models_params_counter.csv", "w") as file:
    file.write('model_name, num_params, ssim, tssim, psnrb,')
    
    for key, value in hps[0].values.items():
        file.write(str(key)+',')
        
    file.write('\n')


for hp in hps:
    model = tuner.hypermodel.build(hp)

    model.fit(cifar_tiny.x_train, cifar_tiny.y_train, epochs=5, validation_data=0.1)

    loss, ssim, tssim, psnrb = model.evaluate(cifar_tiny.x_test, cifar_tiny.y_test)

    with open("models/models_params_counter.csv", "a") as file:
        file.write(model.model_name+','+model.count_params()+','+','+ssim+','+tssim+','+psnrb+',')

        for key, value in hp.values.items():
            file.write(str(value)+',')
        
        file.write('\n')

