import sys, os

from tensorflow.python.keras.engine import training

import numpy as np
import tensorflow as tf
from tensorflow._api.v2.image import rgb_to_grayscale
from io import BytesIO
from PIL import Image
from tensorflow.keras.datasets.cifar10 import load_data as cifar10_load
from TensorflowUtils.DataSet import DataSetABC
from tensorflow.keras.models import Model

from misc import get_model


class DataMod ():
    """
    Essa classe recebe um dataset (imagens no formato np array de preferência), através do construtor, \n 
    com dimensões (num_images, linhas, colunas, rbg) e aplica modificações diversas através de seus métodos. \n

    imports necessários para o funcionamento da classe: \n
        import numpy as np \n
        from tensorflow._api.v2.image import rgb_to_grayscale \n
        from io import BytesIO \n
        from PIL import Image \n

    """
    def __init__ (self, dataSet):
        self.dataSet = dataSet
    

    def rbg_to_gray (self):
        '''
        Coverte as imagens no formato rbg para escala de cinza.
        '''        
        self.dataSet = np.array(rgb_to_grayscale(self.dataSet), dtype='uint8')


    def add_uniform_noise (self, max_pixel_var):
        '''
        Adiciona ruído nas imagens (ruído único para cada imagem), onde valor max_pixel_var é a variação \n
        máxima que um pixel pode sofrer na imagem. 
        '''
        rng = np.random.Generator(np.random.PCG64(12345))
        self.dataSet = np.array(self.dataSet, dtype=int)
        NumImages, Dim1, Dim2, Dim3 = self.dataSet.shape
        
        for image in range(NumImages):
            noiseMatrix = rng.integers(-max_pixel_var, max_pixel_var + 1, (Dim1, Dim2, Dim3), dtype=int)
            self.dataSet[image] = noiseMatrix + self.dataSet[image]
        
        self.dataSet = np.clip(self.dataSet, 0, 255)

        self.dataSet = np.array(self.dataSet, dtype='uint8')


    def add_jpeg_compression_to_grayscale (self, compress_quality):
        """
        Adiciona efeitos da compressão jpeg no dataSet (em escala de cinza) \n
        "compress_quality é o valor da qualidade da compressão (quanto maior, melhor a qualidade, e menor a compressão dos dados)"
        """
        for idx in range(self.dataSet.shape[0]):
            buffer = BytesIO()
            img = Image.fromarray(self.dataSet[idx].reshape(32,32), mode="L")
            img.save(buffer, "JPEG", quality=compress_quality)
            image = Image.open(buffer)
            image = np.asarray(image)
            self.dataSet[idx] = image.reshape(32,32,1)
            buffer.close()
        
        self.dataSet = np.array(self.dataSet, dtype = 'uint8')

    def normalize_dataset(self):
        """
        
        
        """
        pass

class DataSet (DataSetABC):
    """
    A classe guarda os dados para o treino das redes neurais.
    Os datasets podem ser passados pelo construtor ou carregados pelos metodos correspondentes

    ### Datasets disponiveis:
        * rafael_tinyImagenet
        * rafael_cifar_10
        * cifar-10 com ruido uniforme e compressão jpeg.
    """
    def __init__(self, 
                dataset_name: str = "",
                x_train: np.array = [],
                x_test: np.array = [], 
                y_train: np.array = [],
                y_test: np.array = []):
        """
        
        """
        self.name: str = dataset_name
        self.description: str = "None"
        self.parameters: dict = {}
        self.x_train: np.array  = x_train
        self.x_test: np.array  = x_test
        self.y_train: np.array  = y_train
        self.y_test: np.array  = y_test

    def load_cifar_10_with_uniform_noise (self, jpeg_compress_quality, max_pixel_var):
        self.name = "cf10_with_noise|jpeg_qual=" + str(jpeg_compress_quality) + "|max_pixel_var = " +str(max_pixel_var)
        (self.x_train, self.y_train),(self.x_test, self.y_test) = cifar10_load()
        
        dataMod_obj = DataMod(self.x_train)
        dataMod_obj.rbg_to_gray()
        dataMod_obj.add_jpeg_compression_to_grayscale(jpeg_compress_quality)
        dataMod_obj.add_uniform_noise(max_pixel_var)
        self.x_train = dataMod_obj.dataSet

        dataMod_obj.dataSet = self.x_test
        dataMod_obj.rbg_to_gray()
        dataMod_obj.add_jpeg_compression_to_grayscale(jpeg_compress_quality)
        dataMod_obj.add_uniform_noise(max_pixel_var)
        self.x_test = dataMod_obj.dataSet

        dataMod_obj.dataSet = self.y_train
        dataMod_obj.rbg_to_gray()
        self.y_train = dataMod_obj.dataset

        dataMod_obj.dataSet = self.y_test
        dataMod_obj.rbg_to_gray()
        self.y_test = dataMod_obj.dataset

        self.description("Cifar 10 data set with jpeg compression and uniform distribution noise")
        
        self.parameters = {'jpeg_compress_quality' : jpeg_compress_quality,
                            'max_pixel_var' : max_pixel_var}

        return self


    def load_rafael_cifar_10_noise_data (self):
        self.name = "rafael_cifar_10"
        self.description = "Cifar 10 with bad quality"
        self.x_train = np.load("/home/rafaeltadeu/old/autoencoder/X_64x64_treino.npy")
        self.x_test = np.load("/home/rafaeltadeu/old/autoencoder/X_64x64_teste.npy")
        self.y_train = np.load("/home/rafaeltadeu/old/autoencoder/Y_64x64_treino.npy")
        self.y_test = np.load("/home/rafaeltadeu/old/autoencoder/Y_64x64_teste.npy")

        self.x_test = self.x_test.astype('float64')
        self.x_train = self.x_train.astype('float64')
        self.y_test = self.y_test.astype('float64')
        self.y_train = self.y_train.astype('float64')
        
        return self

    def load_rafael_tinyImagenet_64x64_noise_data (self):
        self.name = "rafael_tinyImagenet"
        self.description = "tinyImagenet with bad quality"
        self.x_train = np.load("/home/rafaeltadeu/old/autoencoder/X_tinyImagenet_64x64_treino.npy")
        self.x_test = np.load("/home/rafaeltadeu/old/autoencoder/X_tinyImagenet_64x64_teste.npy")
        self.y_train = np.load("/home/rafaeltadeu/old/autoencoder/Y_tinyImagenet_64x64_treino.npy")
        self.y_test = np.load("/home/rafaeltadeu/old/autoencoder/Y_tinyImagenet_64x64_teste.npy")
        
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train.astype('float32')
        self.y_test = self.y_test.astype('float32')
        self.y_train = self.y_train.astype('float32')

        return self

    def load_discriminator_training_set(self, training_idx:int = None, generator_name: Model = None, custom_objs: dict = None, dataset = 'rafael_cifar_10'):
        """
        
        """
        print("Loading discriminator training set")

        if generator_name == None and training_idx == None:
            raise Exception("No model name or training_idx")

        if generator_name:
            generator = get_model(model_json_name = generator_name)

        if training_idx != None:
            generator = get_model(training_idx = training_idx, custom_objects = custom_objs)

        self.load_by_name(dataset)

        self.name = 'discriminator_training_set'
        self.description = "images with label 1 and 0 for real and fake imgs respectively"
        self.parameters = {'training_idx' : training_idx, 'generator_name' : generator_name, 'custom_objs' : custom_objs, 'dataset' : dataset}

        self.x_train = np.concatenate((self.y_train, generator.predict(self.x_train)))
        self.y_train = np.concatenate((np.ones(shape=(self.y_train.shape[0], 1)), np.zeros(shape = (self.y_train.shape[0], 1))))

        self.x_test = np.concatenate((self.y_test, generator.predict(self.x_test)))
        self.y_test = np.concatenate((np.ones(shape= (self.y_test.shape[0], 1)), np.zeros(shape= (self.y_test.shape[0],1))))

        assert self.x_train.shape[0] == self.y_train.shape[0]
        assert self.x_test.shape[0] == self.y_test.shape[0]
        
        '''
        idx = tf.range(start=0, limit=tf.shape(self.x_train)[0], dtype=tf.int32)

        idx = tf.random.shuffle(idx)

        self.x_train = tf.gather(self.x_train, idx) 
        self.y_train = tf.gather(self.y_train, idx)

        idx = tf.range(start=0, limit=tf.shape(self.x_test)[0], dtype=tf.int32)

        idx = tf.random.shuffle(idx)

        self.x_test = tf.gather(self.x_test, idx) 
        self.y_test = tf.gather(self.y_test, idx)
        '''
        return self

    def concatenateDataSets(self, dataset1, dataset2):
        """
        Concatenate two datasets into a new dataset
        if suceeds, returns 1, else returns 0
        """

        self.name = dataset1.name + "_" + dataset2.name
        self.description = "Concatenation of " + dataset1.name + " and " + dataset2.name + " datasets"
        
        try:
            self.x_train = np.concatenate((dataset1.x_train, dataset2.x_train))
            self.y_train = np.concatenate((dataset1.y_train, dataset2.y_train))
            self.x_test = np.concatenate((dataset1.x_test, dataset2.x_test))
            self.y_test = np.concatenate((dataset1.y_test, dataset2.y_test))
        except:
            print("Error concatenating datasets")
            return 0
    
        return self

    def load_by_name(self, name:str, kwargs: dict = {}):

        if (name == "rafael_tinyImagenet"):
            self.load_rafael_tinyImagenet_64x64_noise_data()
            return self
        
        if (name == "rafael_cifar_10"):
            self.load_rafael_cifar_10_noise_data()
            return self

        if (name == "discriminator_training_set"):
            self.load_discriminator_training_set(**kwargs)

        return self
    
    def get_metadata (self) -> dict:

        return {"name":self.name, "parameters": self.parameters}
    
    def add_gaussian_noise(self, dist_normal : int = 0.1):
        noise = np.random.normal(loc=0.0, scale=dist_normal, size=self.x_train.shape)
        self.x_train = self.x_train + noise

        noise = np.random.normal(loc=0.0, scale=dist_normal, size=self.x_test.shape)
        self.x_test = self.x_test + noise

        return self
