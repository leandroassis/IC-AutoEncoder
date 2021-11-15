import sys, os

from tensorflow.python.eager.monitoring import Metric
sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder'))
sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/modules'))

from abc import ABC, abstractmethod

from DirManager import KerasDirManager
from finding_best_sigma import find_best_sigma_for_ssim

from tensorflow.keras.models import Model, load_model
from pandas import read_csv, DataFrame
from scipy.ndimage import gaussian_filter

import tensorflow as tf
import random as rd
import numpy as np


class KerasTrainingData ():
    """
        Resposible to get training results data to be analised 
    """
    
    def __init__(self) -> None:
        pass
        
        
    
    def get_csv_training_history (self) -> DataFrame:
        """

        """
        dataframe = read_csv(self.csv_pathname, sep=';')
        return dataframe


    def get_best_results(self, metrics_names:list = ['loss', 'ssim_metric', 'psnr_metric'],
                        best = [min, max, max],
                        validation = True,
                        last_results = True) -> dict:
        """
            Function that get the best results from the actual training

            Parameters
            ----------

            metric_names:
                Metric names on the list csv training history (dont include val_*, if vallidation = true)
            
            best:
                For each metric, the function that get the best results in the list

            validation:
                bool telling if validation is considered

            last_results
                bool that includes the last results in the return

            Returns
            -------

            A `dict` that cotains all names and results

        """

        results = {}

        dataframe: DataFrame = self.get_csv_training_history ()

        for metric, func in (metrics_names, best):
            results[f"best_{metric}"] = func(dataframe[metric])
            results[f"best_{metric}_epoch"] = dataframe.loc[dataframe[metric] == results[f"best_{metric}"]]['epoch'][0]

            if validation:
                results[f"best_val_{metric}"] = func(dataframe[f"val_{metric}"])
                results[f"best_val_{metric}_epoch"] = dataframe.loc[dataframe[metric] == results[f"best_val_{metric}"]]['epoch'][0]

        if last_results:
            for metric in metrics_names:
                results[f"last_{metric}"] = dataframe[metric].tolist()[-1]
        
            results['last_epoch'] = dataframe['epoch'].tolist[-1]

        
        return results


    def get_example_imgs (self, model: Model, num_imgs = 4, seed = 12321) -> tf.Tensor :

        rd.seed(seed)

        selected_imgs = rd.sample( range( 0, len(self.dataset.x_test) ), num_imgs )

        input_imgs = []
        output_imgs = []
        expected_imgs = []
        gaussian_imgs = []

        sigma = find_best_sigma_for_ssim(self.dataset.x_test[0:200], self.dataset.y_test[0:200])

        input_imgs = np.array([self.dataset.x_test[idx] for idx in selected_imgs])
        output_imgs = np.array(np.clip(model.predict(input_imgs), a_max= 255, a_min = 0), dtype='uint8')
        expected_imgs = np.array([self.dataset.y_test[idx] for idx in selected_imgs], dtype='uint8')
        gaussian_imgs = gaussian_filter(input_imgs, sigma=(0, sigma, sigma, 0))

        return input_imgs, output_imgs, expected_imgs, gaussian_imgs