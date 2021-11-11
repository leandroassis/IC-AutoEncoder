
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from os import environ
import sys
sys.path.insert(1, '/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/modules/')
sys.path.insert(1, '/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder')
from NeuralNetData import KerasNeuralNetData

import unittest

environ["CUDA_VISIBLE_DEVICES"]="1"

class TestKerasNeuralNetData (unittest.TestCase):

    def test_paralel_segments(self):

        obj = KerasNeuralNetData(model_name='Unet2.1-64x64.json', load_model=True)

        self.assertEqual(obj.number_of_paralel_segments['number'], 4)


    def test_distribution(self):

        inputs = tf.keras.layers.Input(shape=(3,3,1))
        node1 = tf.keras.layers.Conv2D(filters = 200, kernel_size=(3,3), padding='same', kernel_regularizer='L1', bias_regularizer='L2')(inputs)
        node2 = tf.keras.layers.Conv2D(filters = 20, kernel_size=(3,3), padding='same', kernel_regularizer='L1', activity_regularizer='L2')(node1)
        node3 = tf.keras.layers.Conv2D(filters = 20, kernel_size=(3,3), padding='same', kernel_regularizer='L1', bias_regularizer='L2')(node2)
        node4 = tf.keras.layers.Conv2D(filters = 20, kernel_size=(3,3), padding='same', kernel_regularizer='L1', activity_regularizer='L2')(node3)

        model = Model(inputs= inputs, outputs=node4)

        obj = KerasNeuralNetData(model= model)

        self.assertEqual(obj.regularizer_L1_map['kernel'],0.625)
        self.assertEqual(obj.regularizer_map['kernel'],0.625)
        self.assertEqual(obj.regularizer_L2_map['bias'], 0.5)
        self.assertEqual(obj.regularizer_map['bias'], 0.5)
        self.assertEqual(obj.regularizer_L2_map['activity'], (2+4)/(4*2))
        self.assertEqual(obj.regularizer_map['activity'], (2+4)/(4*2))
        
        self.assertLess(obj.parameters_distribution, 0.5)

