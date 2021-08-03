import numpy as np
from tensorflow._api.v2.image import rgb_to_grayscale
from io import BytesIO
from PIL import Image
from tensorflow.keras.datasets.cifar10 import load_data as cifar10_load

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



class DataSet ():
    """
    A classe guarda os dados para o treino das redes neurais.
    Os datasets podem ser passados pelo construtor ou carregados pelos metodos correspondentes

    ### Datasets disponiveis:
        * rafael_tinyImagenet
        * rafael_cifar_10
        * cifar-10 com ruido uniforme e compressão jpeg.
    """
    def __init__(self, dataset_name = "", x_train = [], x_test = [], y_train = [], y_test = []):
        self.name = dataset_name
        self.description = "descrição não introduzida"
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

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


    def load_rafael_cifar_10_noise_data (self):
        self.name = "rafael_cifar_10"
        self.x_train = np.load("/home/rafaeltadeu/autoencoder/X_64x64_treino.npy")
        self.x_test = np.load("/home/rafaeltadeu/autoencoder/X_64x64_teste.npy")
        self.y_train = np.load("/home/rafaeltadeu/autoencoder/Y_64x64_treino.npy")
        self.y_test = np.load("/home/rafaeltadeu/autoencoder/Y_64x64_teste.npy")

    def add_description_to_dataset (self, description_text):
        self.description = description_text

    def load_rafael_tinyImagenet_64x64_noise_data (self):
        self.name = "rafael_tinyImagenet"
        self.x_train = np.load("/home/rafaeltadeu/autoencoder/X_tinyImagenet_64x64_treino.npy")
        self.x_test = np.load("/home/rafaeltadeu/autoencoder/X_tinyImagenet_64x64_teste.npy")
        self.y_train = np.load("/home/rafaeltadeu/autoencoder/Y_tinyImagenet_64x64_treino.npy")
        self.y_test = np.load("/home/rafaeltadeu/autoencoder/Y_tinyImagenet_64x64_teste.npy")
        
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train.astype('float32')
        self.y_test = self.y_test.astype('float32')
        self.y_train = self.y_train.astype('float32')