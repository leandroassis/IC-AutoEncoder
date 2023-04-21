"""
Description
===========

The module contains functions that help in some process of a class method, or something similar.

"""
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.losses import Loss, Reduction, MeanAbsoluteError, binary_crossentropy
from tensorflow.keras.layers import Layer
from tensorflow._api.v2.image import ssim
from datetime import datetime as dt

#from sewar import psnrb
import tensorflow as tf
import numpy as np
from glob import glob

import math
from tensorflow._api.v2.signal import dct, idct
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.ops.tensor_array_ops import TensorArray
    


class DCTL2D (Layer):
    """
    
    
    """
    
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):

        super().__init__(trainable, name, dtype, dynamic, **kwargs)


    def call(self, inputs, **kwargs):

        op1 = tf.matmul(self.C, inputs)
        output = tf.matmul(op1, self.C, transpose_b = True)

        return output

    def build(self, input_shape):

        assert input_shape[-2:] == (8,8)

        C = np.empty(shape = (8,8), dtype = float)

        for j in range(0,8):
            C[0,j] = np.sqrt(8)

        for i in range(1, 8):
            for j in range(0,8):
                C[i,j] = 0.5*tf.math.cos((2*j+1)*i*math.pi/16)

        self.C = tf.constant(C, dtype=tf.float32)

        return super(DCTL2D, self).build(input_shape)


class IDCT2D (Layer):
    """
    
    
    """
    
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):


        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    pass

    
    def call(self, inputs, **kwargs):

        op1 = tf.matmul(self.inverse_C, inputs)
        output = tf.matmul(op1, self.inverse_C, transpose_b = True)

        return output

    def build(self, input_shape):

        C = np.empty(shape = (8,8), dtype = float)

        for j in range(0,8):
            C[0,j] = np.sqrt(8)

        for i in range(1, 8):
            for j in range(0,8):
                C[i,j] = 0.5*tf.math.cos((2*j+1)*i*math.pi/16)

        self.inverse_C = tf.linalg.inv(tf.constant(C, dtype=tf.float32))

        return 


def ssim_metric (y_true,y_pred, max_val = 255, filter_size = 9, filter_sigma = 1.5, k1=0.01, k2=0.03):
    return ssim(y_true, y_pred, max_val = max_val,
                      filter_size = filter_size,
                      filter_sigma = filter_sigma,
                      k1 = k1,
                      k2 = k2)


def psnrb_metric (y_true,y_pred):
    result = []
    
    x = tf.make_ndarray(y_pred)
    y = tf.make_ndarray(y_true)
    

    for idx in range(y_true.shape[0]):
        result.append(psnrb(x[idx], y[idx]))
    return result



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
                raise Exception(f"{path.__len__()} files, found with the name {model_json_name}")

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
