import numpy as np

class NoiseGen ():
    
    def __init__(self):

        self.dataSet = []

    def grayImgSqrNoiseMkr(self, imageDataSet, xRange, yRange, xInit, yInit):

        rng = np.random.default_rng()
        self.dataSet = imageDataSet.copy()
        pixel = 0
        delta = 0 
        NumImages, Dim1, Dim2, Dim3 = imageDataSet.shape
        
        for image in range(NumImages):
            for linha in range(yRange):
                for coluna in range(xRange):
                    pixel = imageDataSet[image][linha+yInit][coluna+xInit][0]
                    randonNumber = rng.standard_normal(1)
                    delta = 0
                    if (randonNumber > 0 and pixel != 255):
                        delta = (1000*randonNumber)%(255-pixel)
                    elif (pixel !=0):
                        delta = (1000*randonNumber)%(pixel)
            
                    self.dataSet[image][linha+yInit][coluna+xInit][0] =+ np.trunc(0.1*delta)

                
    def rGrayImgSqrNoiseMkr(self, imageDataSet, maxRange):

        rng = np.random.default_rng()
        self.dataSet = imageDataSet.copy()
        pixel = 0
        delta = 0 
        NumImages, Dim1, Dim2, Dim3 = imageDataSet.shape

        for image in range(NumImages):
            
            yInit = int(np.trunc((1000*rng.standard_normal())%30))
            xInit = int(np.trunc((1000*rng.standard_normal())%30))
            xRange = int(np.trunc((1000*rng.standard_normal())%(31-xInit)))
            yRange = int(np.trunc((1000*rng.standard_normal())%(31-yInit)))

            for linha in range(yRange):
                for coluna in range(xRange):
                    pixel = imageDataSet[image][linha+yInit][coluna+xInit][0]
                    randonNumber = rng.standard_normal(1)
                    delta = 0
                    if (randonNumber > 0 and pixel != 255):
                        delta = (1000*randonNumber)%(255-pixel)
                    elif (pixel !=0):
                        delta = (1000*randonNumber)%(pixel)
            
                    self.dataSet[image][linha+yInit][coluna+xInit][0] =+ np.trunc(delta)
        

    def grayLowNoiseMkr(self, imageDataSet, alpha):

        rng = np.random.Generator(np.random.PCG64(12345))
        self.dataSet = imageDataSet.copy()
        self.dataSet = np.array(self.dataSet, dtype=int)
        NumImages, Dim1, Dim2, Dim3 = imageDataSet.shape
        
        for image in range(NumImages):
            noiseMatrix = rng.integers(-alpha,alpha, (Dim1, Dim2, Dim3), dtype=int)
            self.dataSet[image] = noiseMatrix + self.dataSet[image]
        for image in range(NumImages):
            for linha in range(Dim1):
                for coluna in range (Dim2):
                    self.dataSet[image][linha][coluna][0] = min(255, self.dataSet[image][linha][coluna][0])
                    self.dataSet[image][linha][coluna][0] = max(0, self.dataSet[image][linha][coluna][0])

        self.dataSet = np.array(self.dataSet, dtype='uint8')


        
