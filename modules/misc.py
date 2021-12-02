"""
Description
===========

The module contains functions that help in some process of a class method, or something similar.

"""
from tensorflow.keras.models import Model
import json
from tensorflow._api.v2.image import ssim
from datetime import datetime as dt
from tensorflow.keras.losses import Loss, Reduction, MeanAbsoluteError
from sewar import psnrb
from tensorflow import keras
from typing import List
import tensorflow as tf
import numpy as np

from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.ops.tensor_array_ops import TensorArray


class LSSIM (Loss):

    def __init__(self, max_val = 255, filter_size=9, filter_sigma=1.5, k1=0.01, k2=0.03, name = "LSSIM", reduction = Reduction.AUTO) -> None:
        
        super(LSSIM, self).__init__(name = name, reduction = reduction)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

    
    def call (self,y_true,y_pred):
        return 1-ssim(y_true, y_pred, max_val = self.max_val,
                      filter_size = self.filter_size,
                      filter_sigma = self.filter_sigma,
                      k1 = self.k1,
                      k2 = self.k2)


class AdversarialLoss(Loss):

    def __init__(self, adversarial_model: Model, model_path, reduction=Reduction.AUTO, name=None):
        self.name = adversarial_model.name
        super().__init__(reduction=reduction, name=name)

        if isinstance(adversarial_model, str):
            self.adversarial_model = load_model(f"{model_path}/{adversarial_model}", compile = False)
        elif issubclass(adversarial_model, Model):
            self.adversarial_model = adversarial_model
        else:
            raise Exception("Invalid model passed")

    def call(self, y_true, y_pred):
        return self.adversarial_model.predict(y_pred)


class LossLinearCombination (Loss):

    def __init__(self, losses: List[Loss], weights: list = None, bias_vector:list = None, name: str = "", reduction = Reduction.AUTO) -> None:
        standard_name = ''
        for loss in losses:
            standard_name += f"-{loss().name}"
        standard_name = standard_name[1:]
        super(LossLinearCombination, self).__init__(name = standard_name, reduction = reduction)
        self.losses = losses
        self.weights = weights
        self.bias_vector = bias_vector

        if not self.weights:
            self.weights = tf.ones(shape=losses.__len__())

        if not self.bias_vector:
            self.bias_vector = tf.zeros(shape=losses.__len__())

    def call(self, y_true,y_pred):
        result = 0
        step1 = 0
        step2 = 0

        for loss, weight, bias in zip(self.losses, self.weights, self.bias_vector):
            step1 += weight*loss().call(y_true, y_pred)
            step2 +=  bias*tf.ones(shape=step1.shape)
            
        return step1 + step2


def ssim_metric (y_true,y_pred, max_val = 255, filter_size = 9, filter_sigma = 1.5, k1=0.01, k2=0.03):
    return ssim(y_true, y_pred, max_val = max_val,
                      filter_size = filter_size,
                      filter_sigma = filter_sigma,
                      k1 = k1,
                      k2 = k2)


def psnrb_metric (y_true,y_pred):
    if len(y_true.shape) == 4:
        result = []
        for idx in range(y_true.shape[0]):
            result.append(psnrb (y_true[idx], y_pred[idx]))
        return result

    return psnrb (y_true, y_pred)



def get_current_time_and_data ():
    '''
        Retorna o tempo em horas:minutos, e data em ano-mes-dia (str)
    '''
    current_datetime = dt.now()
    time = current_datetime.time()
    date = current_datetime.date()
    date_str = str(date)
    time_str = time.strftime('%X')[:5]
    return date_str, time_str


def get_last_epoch (csv_pathname):
    """
        Retorna a ultima época treinada de um checkpoint. \n
        obs: retorna -1 quando nenhum treino foi realizado para o treino inciar na época 0.
    """
    try:
        file = open(csv_pathname, 'r')
    except FileNotFoundError:
        return -1
    lines = file.read().splitlines()
    if (len(lines) == 0):
        return -1
    file.close()
    last_line = lines[-1]
    last_epoch = int(last_line.split(';')[0])
    return last_epoch

