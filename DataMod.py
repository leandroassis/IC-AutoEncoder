import numpy as np
from tensorflow._api.v2.image import rgb_to_grayscale
from io import BytesIO
from PIL import Image

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
        self.dataSet = rgb_to_grayscale(self.dataSet)


    def add_standard_Noise (self, max_pixel_var):
        '''
        Adiciona ruído nas imagens (ruído único para cada imagem), onde valor max_pixel_var é a variação \n
        máxima que um pixel pode sofrer na imagem. 
        '''
        rng = np.random.Generator(np.random.PCG64(12345))
        self.dataSet = np.array(self.dataSet, dtype=int)
        NumImages, Dim1, Dim2, Dim3 = self.dataSet.shape
        
        for image in range(NumImages):
            noiseMatrix = rng.integers(-max_pixel_var, max_pixel_var, (Dim1, Dim2, Dim3), dtype=int)
            self.dataSet[image] = noiseMatrix + self.dataSet[image]
        
        self.dataSet = np.clip(self.dataSet, 0, 255)

        self.dataSet = np.array(self.dataSet, dtype='uint8')


    def add_jpeg_compression_to_grayscale (self, compress_quality):
        """
        Adiciona efeitos da compressão jpeg no dataSet \n
        "compress_quality é o valor da qualidade da compressão (quanto maior, melhor a qualidade, e menor a compressão dos dados)"
        """
        for idx in range(self.dataSet.shape[0]):
            buffer = BytesIO()
            img = Image.fromarray(self.dataSet[idx].reshape(32,32), mode="L")
            img.save(buffer, "JPEG", quality=compress_quality)
            image = Image.open(buffer)
            image = np.asarray(image)
            self.dataSet[idx] = image
            buffer.close()


        
