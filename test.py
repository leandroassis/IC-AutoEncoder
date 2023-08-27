from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3"

from modules.DataMod import DataSet
from run1.analyze import plot_model_graphic

tinyDataSet = DataSet()

tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
print(tinyDataSet.x_test[15])

tinyDataSet = tinyDataSet.add_gaussian_noise()

print(tinyDataSet.x_test[15])


plot_model_graphic(dataset=tinyDataSet, magic_number=[1, 2, 3, 4, 5], output_path="/src/run2/noise.png")