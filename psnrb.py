import math
import tensorflow as tf
from os import environ
environ["CUDA_VISIBLE_DEVICES"]="3"
import sewar
import numpy as np


def blocking_efect_factor (im: tf.Tensor, block_size = 8) -> tf.Tensor:
	"""

	"""
	n_imgs, height, width, channels = im.shape

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, :, i] - im[:, :, i+1]
		d_b += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, :, i] - im[:, :, i+1]
		d_bc += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# v_b for loop
	for j in list(v_b):
		diff = im[:, j, :] - im[:, j+1, :]
		d_b += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# V_bc for loop
	for j in list(v_bc):
		diff = im[:, j, :] - im[:, j+1, :]
		d_bc += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	t = np.log2(block_size)/np.log2(min(height, width))
	
	# BEF
	bef = t*(d_b - d_bc)

	return tf.math.maximum(bef, tf.zeros(bef.shape, dtype = bef.dtype))



def psnrb (target_imgs: tf.Tensor, degraded_imgs: tf.Tensor, ) -> tf.float32:
	"""
	Computes the PSNR-B for a batch of images

	### Obs: 
	The order of images are important, 'couse of the block efect factor (BEF) calculation depends only of one image.

	### Ref:
	Quality Assessment of Deblocked Images: Changhoon Yim, Member, IEEE, and Alan Conrad Bovik, Fellow, IEEE 
	"""
	imgs_shape = degraded_imgs.shape.__len__()

	assert imgs_shape == 4

	img_mse = tf.reduce_mean(tf.square(target_imgs - degraded_imgs), axis=[1,2,3])

	bef_total = blocking_efect_factor(degraded_imgs)

	psnr_b =  tf.math.add(-10*tf.math.log(bef_total + img_mse)/math.log(10), 10*math.log(255**2, 10))

	return psnr_b


