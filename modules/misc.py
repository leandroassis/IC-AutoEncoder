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

from sewar import psnrb
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




def _compute_bef(im, block_size=8):
	"""Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image

	C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
		vol. 20, no. 1, pp. 88-98, Jan. 2011.

	:param im: input image (numpy ndarray)
	:param block_size: Size of the block over which DCT was performed during compression
	:return: float -- bef.
	"""
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = tf.array(range(0, width - 1))
	h_b = tf.array(range(block_size - 1, width - 1, block_size))
	h_bc = tf.array(list(set(h).symmetric_difference(h_b)))

	v = tf.array(range(0, height - 1))
	v_b = tf.array(range(block_size - 1, height - 1, block_size))
	v_bc = tf.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, i] - im[:, i+1]
		d_b += tf.sum(tf.square(diff))

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, i] - im[:, i+1]
		d_bc += tf.sum(tf.square(diff))

	# v_b for loop
	for j in list(v_b):
		diff = im[j, :] - im[j+1, :]
		d_b += tf.sum(tf.square(diff))

	# V_bc for loop
	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += tf.sum(tf.square(diff))

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	if d_b > d_bc:
		t = tf.math.log(block_size)/tf.math.log(min(height, width))
	else:
		t = 0

	# BEF
	bef = t*(d_b - d_bc)

	return bef


def psnrb(GT, P):
	"""Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)

	:param GT: first (original) input image in YCbCr format or Grayscale.
	:param P: second (corrected) input image in YCbCr format or Grayscale..
	:return: float -- psnr_b.
	"""
	if len(GT.shape) == 3:
		GT = GT[:, :, 0]

	if len(P.shape) == 3:
		P = P[:, :, 0]

	imdff = tf.double(GT) - tf.double(P)

	mse = tf.mean(tf.square(imdff.flatten()))
	bef = _compute_bef(P)
	mse_b = mse + bef

	if tf.amax(P) > 2:
		psnr_b = 10 * tf.math.log(255**2/mse_b)/tf.math.log(10)
	else:
		psnr_b = 10 * tf.math.log(1/mse_b)/tf.math.log(10)

	return psnr_b
