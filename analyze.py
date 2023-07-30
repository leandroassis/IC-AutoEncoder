from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3"

from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM, LPSNRB, L3SSIM
from modules.misc import ssim_metric
from modules.ImageMetrics.metrics import three_ssim, psnrb
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import random as rd

from keras import models 

tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()

tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()

# concatenates the datasets
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)

def plot_model_graphic(model, dataset, output_path):
        # plots the model

        plt.figure(figsize=(10, 10))

        columns = 3
        rows = 5

        for idx in range(rows):
                magic_number = rd.randint(0, len(dataset.x_test) - 1)

                plt.subplot(rows, columns, columns*idx + 1)
                plt.title("Original Image")
                plt.imshow(dataset.x_test[magic_number])
                plt.subplot(rows, columns, columns*idx + 2)
                plt.title("Goal Image")
                plt.imshow(dataset.y_test[magic_number])
                plt.subplot(rows, columns, columns*idx + 3)
                plt.title("Predicted Image")
                plt.imshow(model.predict(dataset.x_test[magic_number][0]))

        plt.savefig(output_path)
        plt.close()

NNmodels = {}

for path in ["AutoEncoder-2.3-64x64.json", "ResidualAutoEncoder-0.1-64x64.json", "Unet2.3-64x64.json"]:
        # reads the model
        with open("models/arch/"+path, "r") as json_file:
                model = models.model_from_json(json_file.read())
                NNmodels[model.name] = model


for model in NNmodels:
    for (dirpath, dirnames, filenames) in walk("logs/run1/weights/"):
        for filename in filenames:
            if filename.startswith(model):

                NNmodels[model].load_weights("logs/run1/weights/"+filename)
                loss = LSSIM() if "LSSIM" in filename else LPSNRB() if "LPSNRB" in filename else L3SSIM() if "L3SSIM" in filename else LSSIM()
                NNmodels[model].compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = loss, metrics = [ssim_metric, three_ssim, psnrb])

                for dataset in [cifarAndTinyDataSet, cifarDataSet, tinyDataSet]:
                    if dataset.name in filename:
                        try:
                            loss, ssim, tssim, psnrb = NNmodels[model].evaluate(x = dataset.x_test, y = dataset.y_test)
                        except:
                            print("Error evaluating model: " + filename.split(".h5")[0])
                            print("\n")
                        else:
                            with open("logs/run1/metrics/results.csv", "a") as results:
                                results.write(filename.split(".h5")[0] + "," + str(loss) + "," + str(ssim) + "," + str(tssim) + "," + str(psnrb) + "\n")

                        plot_model_graphic(NNmodels[model], dataset, "logs/run1/plots/"+filename.split(".h5")[0]+".png")

