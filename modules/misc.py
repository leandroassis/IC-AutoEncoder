"""
Description
===========

The module contains functions that help in some process of a class method, or something similar.

"""
from genericpath import isdir
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model, model_from_json
import json
from tensorflow._api.v2.image import ssim
from datetime import datetime as dt
from tensorflow.keras.losses import Loss, Reduction, MeanAbsoluteError, binary_crossentropy
from sewar import psnrb
from tensorflow import keras
from typing import List
import tensorflow as tf
import numpy as np
from glob import glob

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

    def __init__(self, training_idx: int = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)
    
    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        return binary_crossentropy(y_pred = self.adversarial_model(y_pred), y_true = tf.ones(shape= (y_pred.shape[0], 1)))


class L1_AdversarialLoss(Loss):
    def __init__(self, training_idx: int = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'L1_AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)

    def call(self, y_true, y_pred):
        return binary_crossentropy(tf.ones(shape= (y_pred.shape[0], 1)), self.adversarial_model(y_pred)) + tf.keras.losses.mean_absolute_error(y_true, y_pred)


class LSSIM_AdversarialLoss(Loss):
    def __init__(self, training_idx: int = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'LSSIM_AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)

    def call(self, y_true, y_pred):
        return binary_crossentropy(y_pred = self.adversarial_model(y_pred), y_true = tf.ones(shape= (y_pred.shape[0], 1))) + LSSIM().call(y_true,y_pred)


class LossLinearCombination (Loss):
    """
        Important: If the loss was a "Loss" subclass, it has to be initiated previously.
    
    """
    def __init__(self, losses: List[Loss], weights: list = None, bias_vector:list = None, name: str = "", reduction = Reduction.AUTO) -> None:
        standard_name = ''
        for loss in losses:
            standard_name += f"-{loss.name}"
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
        step1 = 0
        step2 = 0

        for loss, weight, bias in zip(self.losses, self.weights, self.bias_vector):
            if issubclass(loss, Loss):
                step1 += weight*loss.call(y_true, y_pred)
            elif (loss.__class__.__name__ == 'function' or loss.__class__.__name__ == "Function"): # TF Function
                step1 += weight*loss(y_true, y_pred)
                step2 += bias*tf.ones(shape=step1.shape)
            
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
        Returns the current time and date.

        Receives:
            Nothing

        Returns:
            date, time (str) respectively

        Raises:
            Nothing
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



def get_model(training_idx: int = None, custom_objects: dict = None, compile = False, model_json_name: str = None, json_models_path: str = None) -> Model:
    """
    
    """

    if training_idx != None:
        model_save_path = glob(f"logs/**/{training_idx}/model", recursive = True)

        if model_save_path.__len__() != 1:
            raise Exception(f"No training, or multiple trainings, found with the training index {training_idx}. This shouldn't be happening")

        model_save_path = model_save_path[0]

        model = load_model(model_save_path, custom_objects = custom_objects, compile = compile)

        return model

    if model_json_name:

        if not json_models_path:

            path = glob(f"**/{model_json_name}", recursive = True)

            if path.__len__() != 1:
                raise Exception(f"{json_models_path.__len__()} files, found with the name {model_json_name}")

            path = path[0]

        else:
            path = f"{json_models_path}/{model_json_name}"


        with open(path, 'r') as json_file:
                architecture = json_file.read()
                model = model_from_json(architecture)
                json_file.close()

        return model


def get_loss_name(loss):

    if isinstance(loss, list):
        loss_name:str = ""
        for item in loss:
            loss_name += f"{item.__name__}+"
        loss_name = loss_name[:-1]

    elif issubclass(loss, Loss):
        loss_name = loss.__name__

    return loss_name