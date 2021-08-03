from PIL.Image import init
from tensorflow._api.v2.image import ssim
from datetime import datetime as dt
from tensorflow.keras.callbacks import Callback
from collections import OrderedDict
from collections import Iterable

import os
import csv
import numpy as np
import json


def Ssim (i1, i2):
    max_val = 255,
    filter_size = 8
    return -ssim(i1, i2, max_val=max_val, filter_size = filter_size)





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
    last_epoch = int(last_line.split(',')[0])
    return last_epoch


def write_params_log (file_pathname, json_obj):
    file = open(file_pathname, 'w')
    json.dump(json_obj, file, indent=3)
    file.close()

