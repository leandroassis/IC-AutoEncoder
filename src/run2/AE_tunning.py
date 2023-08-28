from keras_tuner import HyperBand

from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3"

from models.functions.autoEncoder import create_AE_model
from modules.DataMod import DataSet

tuner = HyperBand(create_AE_model,
                  objective='val_three_ssim',
                  max_epochs=5,
                  factor=4,
                  directory='my_dir',
                  project_name='AE_tunning')

tuner.search_space_summary()

cifar, tiny, cifar_tiny = DataSet(), DataSet(), DataSet()

cifar = cifar.load_rafael_cifar_10_noise_data()
tiny = tiny.load_rafael_tinyImagenet_64x64_noise_data()

cifar_tiny = cifar_tiny.concatenateDataSets(cifar, tiny)
cifar_tiny = cifar_tiny.add_gaussian_noise(0.3)

#tuner.search(cifar_tiny.x_test, cifar_tiny.y_test, epochs=5, validation_data=0.1)
tuner.search(cifar_tiny.x_train, cifar_tiny.y_train, epochs=5, validation_data=0.1)

hps = tuner.get_best_hyperparameters(20)

print(hps[13])

for hp in hps:
    model = create_AE_model(hp)
    model.fit(cifar_tiny.x_train, cifar_tiny.y_train, epochs=5, validation_data=0.1)

    loss, ssim, tssim, psnrb = model.evaluate(cifar_tiny.x_test, cifar_tiny.y_test)

    with open("models/models_params_counter.csv", "a") as file:
        file.write(model.model_name+','+model.count_params()+','+','+ssim+','+tssim+','+psnrb+'\n')

