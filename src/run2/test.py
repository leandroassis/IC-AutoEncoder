from sys import path
from os import getcwd, environ, walk

path.insert(0, getcwd())
path.insert(0, getcwd() + "/modules/")
environ["CUDA_VISIBLE_DEVICES"] = "3"

from modules.DataMod import DataSet
import matplotlib.pyplot as plt

tinyDataSet = DataSet()

tinyDataSet.load_rafael_tinyImagenet_64x64_noise_data()
print(tinyDataSet.x_test[15])

tinyDataSet = tinyDataSet.add_gaussian_noise(dist_normal=0.3)

print(tinyDataSet.x_test[15])

plt.figure(figsize=(8, 8))

columns = 3
rows = 5

magic_number = [15, 16, 17, 18, 19]

plt.subplot(rows, columns, 1)
plt.title("Noise Image")
plt.subplot(rows, columns, 2)
plt.title("Goal Image")
plt.subplot(rows, columns, 3)
plt.title("Output Image")

for idx in range(rows):
    plt.subplot(rows, columns, columns*idx + 1)
    plt.imshow(tinyDataSet.x_test[magic_number[idx]], cmap="gray")
    plt.axis("off")
    plt.subplot(rows, columns, columns*idx + 2)
    plt.imshow(tinyDataSet.y_test[magic_number[idx]], cmap="gray")
    plt.axis("off")

plt.savefig("ruido.png", dpi=600)
plt.close()