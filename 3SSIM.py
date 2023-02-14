import tensorflow as tf
from os import environ
environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np


def compute_ssim_map (imgs_1, imgs_2) -> tf.Tensor:
    pass

def ssim3 (imgs_1: tf.Tensor, imgs_2: tf.Tensor):
    """
    
    """
    img_grad = tf.image.sobel_edges(imgs_2)
    imgs_magnitude_grad = tf.sqrt( tf.square(img_grad[:,:,:,:, 0]) + tf.square(img_grad[:,:,:,:, 1]))
    


img1_np = np.random.normal(127, 255/3, size = (2,64,64,1))
img2_np = np.random.normal(127, 35/3, size = (2,64,64,1))
img1 = tf.constant(img1_np, dtype="float64")
img2 = tf.constant(img2_np, dtype="float64")