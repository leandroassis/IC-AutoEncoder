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

def plot_model_comparison_graphic(num_sets = 3, num_subplots = 3):
    barWidth = 0.13

    pos_barra = [[x for x in range(num_sets)]]

    for _ in range(num_subplots):
        pos_barra.append([x + barWidth for x in pos_barra[-1]])

    fig, ax = plt.subplots(num_subplots, 1, figsize=(8, 8))
    ax1 = ax.twinx()
    ax2 = ax.twinx()

    plt.setp(ax, xticks=[r + barWidth for r in range(num_sets)], xticklabels=['autEnc', 'Unet', 'resAut'])
    plt.setp(ax, ylabel='ssim', xlabel='Modelos')
    plt.setp(ax1, ylabel='psnr', xlabel='Modelos')
    plt.setp(ax2, ylabel='3ssim', xlabel='Modelos')

    fig.tight_layout(pad=3.0)

    for idx, funcao in enumerate(["cifar-10", "tinyimg", "cifar+tiny"]):
        mediaFlops, stdFlops = resultadosO4.getFLOPSValues(funcao)
        
        ax[0].bar(pos_barra[idx], mediaFlops, width = barWidth, label = funcao)
        ax[0].grid(axis='y', alpha=0.75)
        ax[0].set_title("Loss: LSSIM", fontsize=10)
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), shadow=True, ncol=num_sets)
        
        ax1[0].bar(pos_barra[idx], mediaFlops, width = barWidth, label = funcao)
        ax2[0].bar(pos_barra[idx], mediaFlops, width = barWidth, label = funcao)

    # O3
    for idx, funcao in enumerate(["normal", "avx", "unroll", "block"]):
        mediaFlops, stdFlops = resultadosO3.getFLOPSValues(funcao)

        ax[1].bar(pos_barra[idx], mediaFlops, width = barWidth, label = funcao)
        ax[1].grid(axis='y', alpha=0.75)
        ax[1].set_title("O3", fontsize=10)

    # O0
    for idx, funcao in enumerate(["normal", "avx", "unroll", "block"]):
        mediaFlops, stdFlops = resultadosO0.getFLOPSValues(funcao)

        ax[2].bar(pos_barra[idx], mediaFlops, width = barWidth, label = funcao)
        ax[2].grid(axis='y', alpha=0.75)
        ax[2].set_title("O0", fontsize=10)

     

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
                            loss_r, ssim, tssim, psnrb = NNmodels[model].evaluate(x = dataset.x_test, y = dataset.y_test)
                        except:
                            print("Error evaluating model: " + filename.split(".h5")[0])
                            print("\n")
                        else:
                            with open("logs/run1/metrics/results.csv", "a") as results:
                                results.write(str(model) + "," + str(dataset.name) + "," + str(loss.name) + "," + str(ssim) + "," + str(tssim) + "," + str(psnrb) + "\n")

                        plot_model_graphic(NNmodels[model], dataset, "logs/run1/plots/"+filename.split(".h5")[0]+".png")

