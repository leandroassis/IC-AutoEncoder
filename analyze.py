from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "1"

from modules.DataMod import DataSet
from modules.CustomLosses import LSSIM, LPSNRB, L3SSIM
from modules.misc import ssim_metric
from modules.ImageMetrics.metrics import three_ssim, psnrb
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import random as rd

from keras import models 

import pandas as pd

print("Libraries imported successfully!")

print("Fetching datasets...")
tinyDataSet, cifarDataSet, cifarAndTinyDataSet = DataSet(), DataSet(), DataSet()

tinyDataSet = tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
cifarDataSet = cifarDataSet.load_rafael_cifar_10_noise_data()

# concatenates the datasets
cifarAndTinyDataSet = cifarAndTinyDataSet.concatenateDataSets(cifarDataSet, tinyDataSet)

print("Datasets fetched successfully!")

def get_models_mean_score(loss_name, metric_name):
    results_csv = pd.read_csv("logs/run1/metrics/results.csv")

    mean = []
    std = []
    
    # filter the goal metric results by the model name and dataset name which has the loss name

    for model_name in ["AutoEncoder-2.3-64x64", "Unet2.3-64x64", "ResidualAutoEncoder-0.1-64x64"]:
         for dataset_name in ["rafael_tinyImagenet", "rafael_cifar_10", "rafael_cifar_10_rafael_tinyImagenet"]:
            filtered_results = results_csv[(results_csv["model_name"] == model_name) & (results_csv["loss_name"] == loss_name) & (results_csv["dataset_name"] == dataset_name)]
            mean.append(filtered_results[metric_name].mean())
            std.append(filtered_results[metric_name].std())

    return mean, std

def plot_model_comparison_graphic(num_sets = 9, num_subplots = 3):
    barWidth = 0.15
    pos_barra = [[x for x in range(num_sets)]]

    for _ in range(num_sets-1):
        pos_barra.append([x + barWidth for x in pos_barra[-1]])

    fig, ax = plt.subplots(num_subplots, 1, figsize=(12, 10))
    plt.setp(ax, xticks=[r + barWidth for r in range(num_sets)], xticklabels=['AE+tiny', 'AE+cifar', 'AE+both', 'UN+tiny', 'UN+cifar', 'UN+both', 'RAE+tiny', 'RAE+cifar', 'RAE+both'])
    plt.setp(ax, ylabel='Score')

    ax2 = ax[0].twinx()
    ax3 = ax[0].twinx()
    ax4 = ax[1].twinx()
    ax5 = ax[1].twinx()
    ax6 = ax[2].twinx()
    ax7 = ax[2].twinx()

    # sets the y-axis labels
    ax[0].set_ylabel('ssim')
    ax[1].set_ylabel('ssim')
    ax[2].set_ylabel('ssim')
    ax2.set_ylabel('3ssim')
    ax4.set_ylabel('3ssim')
    ax6.set_ylabel('3ssim')
    ax3.set_ylabel('psnrb')
    ax5.set_ylabel('psnrb')
    ax7.set_ylabel('psnrb')

    # move the last y-axis spine over to the right by 20% of the width of the axes
    ax3.spines['right'].set_position(('outward', 60))
    ax5.spines['right'].set_position(('outward', 60))
    ax7.spines['right'].set_position(('outward', 60))

    fig.tight_layout(pad=3.0)

    color1, color2, color3 = plt.cm.viridis([0, .5, .9])

    ax[0].yaxis.label.set_color(color1)
    ax2.yaxis.label.set_color(color2)
    ax3.yaxis.label.set_color(color3)
    
    ax[1].yaxis.label.set_color(color1)
    ax4.yaxis.label.set_color(color2)
    ax5.yaxis.label.set_color(color3)

    ax[2].yaxis.label.set_color(color1)
    ax6.yaxis.label.set_color(color2)
    ax7.yaxis.label.set_color(color3)


    for idx, metric in enumerate(["ssim", "tssim", "psnrb"]):
        scores, std_scores = get_models_mean_score("LSSIM", metric)

        if metric == "ssim": 
            ax[0].bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color1)
            ax[0].grid(axis='y', alpha=0.75)
            ax[0].set_title("Loss = LSSIM", fontsize=10)
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
        elif metric == "tssim":
            ax2.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color2)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
            ax2.grid(axis='y', alpha=0.75)            
        else:
            ax3.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color3)
            ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
            ax3.grid(axis='y', alpha=0.75)

    for idx, metric in enumerate(["ssim", "tssim", "psnrb"]):
        scores, std_scores = get_models_mean_score("L3SSIM", metric)

        if metric == "ssim": 
            ax[1].bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color1)
            ax[1].grid(axis='y', alpha=0.75)
            ax[1].set_title("Loss = L3SSIM", fontsize=10)
        elif metric == "tssim":
            ax4.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color2)
            ax4.grid(axis='y', alpha=0.75)            
        else:
            ax5.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color3)
            ax5.grid(axis='y', alpha=0.75)

    for idx, metric in enumerate(["ssim", "tssim", "psnrb"]):
        scores, std_scores = get_models_mean_score("LPSNRB", metric)

        if metric == "ssim": 
            ax[2].bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color1)
            ax[2].grid(axis='y', alpha=0.75)
            ax[2].set_title("Loss = LPSNRB", fontsize=10)
        elif metric == "tssim":
            ax6.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color2)
            ax6.grid(axis='y', alpha=0.75)            
        else:
            ax7.bar(pos_barra[idx], scores, width = barWidth, label = metric, color=color3)
            ax7.grid(axis='y', alpha=0.75)

    plt.savefig("logs/run1/plots/model_comparison.png", dpi=600)
    
def plot_model_graphic(model, dataset, output_path):
        # plots the model

        plt.figure(figsize=(12, 12))

        columns = 3
        rows = 5

        plt.subplot(rows, columns, 1)
        plt.title("Noise Image")
        plt.subplot(rows, columns, 2)
        plt.title("Goal Image")
        plt.subplot(rows, columns, 3)
        plt.title("Output Image")

        for idx in range(rows):
                magic_number = rd.randint(0, len(dataset.x_test) - 1)

                plt.subplot(rows, columns, columns*idx + 1)
                plt.imshow(dataset.x_test[magic_number], cmap="gray")
                plt.axis("off")
                plt.subplot(rows, columns, columns*idx + 2)
                plt.imshow(dataset.y_test[magic_number], cmap="gray")
                plt.axis("off")
                plt.subplot(rows, columns, columns*idx + 3)
                plt.imshow(model.predict(dataset.x_test)[magic_number], cmap="gray")
                plt.axis("off")

        plt.savefig(output_path)
        plt.close()

NNmodels = {}


print("Loading models...")
for path in ["AutoEncoder-2.3-64x64.json", "ResidualAutoEncoder-0.1-64x64.json", "Unet2.3-64x64.json"]:
        # reads the model
        with open("models/arch/"+path, "r") as json_file:
                model = models.model_from_json(json_file.read())
                NNmodels[model.name] = model
print("Models loaded!")

print("Inicializing results sheet...")
with open("logs/run1/metrics/results.csv", "w") as results_csv:
     results_csv.write("model_name,loss_name,dataset_name,ssim,tssim,psnrb\n")

print("Results sheet inicialized!")
     

print("Starting models analysis...")
for model in NNmodels:
    for (dirpath, dirnames, filenames) in walk("logs/run1/weights/"):
        for filename in filenames:
            if filename.startswith(model):
                
                print("Loading models weights and compiling  it...")
                NNmodels[model].load_weights("logs/run1/weights/"+filename)
                loss = LSSIM() if "LSSIM" in filename else L3SSIM() if "L3SSIM" in filename else LPSNRB() if "LPSNRB" in filename else LSSIM()
                NNmodels[model].compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False), loss = loss, metrics = [ssim_metric, three_ssim, psnrb])
                print("Weights loaded and model compiled!")

                for dataset in [cifarAndTinyDataSet, cifarDataSet, tinyDataSet]:
                    if dataset.name in filename:
                        try:
                            print("Evaluating model "+filename.split(".h5")[0] + " with dataset "+dataset.name+"...")
                            loss_r, ssim, tssim, psnrb_s = NNmodels[model].evaluate(x = dataset.x_test, y = dataset.y_test)
                        except KeyboardInterrupt:
                             exit()
                        except Exception as e:
                            print("Error evaluating model: " + filename.split(".h5")[0])
                            print(e)
                            print("\n")
                        else:
                            print("Model evaluated!")

                            print("Saving results...")
                            with open("logs/run1/metrics/results.csv", "a") as results_csv:
                                results_csv.write(str(model) + "," + str(loss.name) + "," + str(dataset.name) + "," + str(ssim) + "," + str(tssim) + "," + str(psnrb_s) + "\n")
                            print("Results saved!")
                        
                        print("Generating model graphic...")
                        plot_model_graphic(NNmodels[model], dataset, "logs/run1/plots/"+filename.split(".h5")[0]+".png")
                        print("Model graphic generated!")
                        break

print("Models analysis finished!")

print("Generating model comparison graphic...")
plot_model_comparison_graphic()
print("Model comparison graphic generated!")
