"""
Description
===========

The Misc module contains functions that help in some process of a class method, or something similar.

"""
from tensorflow.keras.models import Model
import json
from tensorflow._api.v2.image import ssim
from datetime import datetime as dt
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.python.ops.image_ops_impl import psnr



def get_neural_net_node_deep(model: Model) -> dict:
    """
        This function wil define the deep of a neuron with the longest path from the initial node in graph network

        receives: 
            kr.models.Model
        returns: 
            `dict` containing {'name':'deep', ...}
        raises: 
            Nothing
    """
    
    layers_config: dict = json.loads(model.to_json())['config']['layers']

    layers_deep: dict = {}

    for layer in layers_config:

        if layer['inbound_nodes'].__len__() == 0:
            layers_deep[layer['name']] = 0

        elif layer['class_name'] == "Concatenate":

            list_of_preveous_layers = [name[0] for name in layer['inbound_nodes'][0]] 

            layers_deep[layer['name']] = max([layers_deep[layer] for layer in list_of_preveous_layers]) + 1

        else:

            layers_deep[layer['name']] = layers_deep[layer['inbound_nodes'][0][0][0]] + 1


    return layers_deep




class LSSIM (Loss):

    def __init__(self, name = "LSSIM", reduction = Reduction.AUTO, max_val = 255, filter_size=9, filter_sigma=1.5, k1=0.01, k2=0.03) -> None:
        
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


def ssim_metric (y_true,y_pred, max_val = 255, filter_size = 9, filter_sigma = 1.5, k1=0.01, k2=0.03):
    return ssim(y_true, y_pred, max_val = max_val,
                      filter_size = filter_size,
                      filter_sigma = filter_sigma,
                      k1 = k1,
                      k2 = k2)


def psnr_metric (y_true,y_pred, max_val = 255):
    return psnr(y_true, y_pred, max_val = max_val)



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



