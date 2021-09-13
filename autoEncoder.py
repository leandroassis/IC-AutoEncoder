from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "AutoEncoder-1.0-64x64"

inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 10, kernel_size = 3, padding = 'same', activation = 'relu')(inputs)
layer_2 = MaxPooling2D()(layer_1)
layer_3 = Conv2D(filters = 20, kernel_size = 3, padding = 'same', activation = 'relu')(layer_2)
layer_4 = MaxPooling2D()(layer_3)
layer_5 = Conv2D(filters = 30, kernel_size = 3, padding = 'same', activation = 'relu')(layer_4)
layer_6 = UpSampling2D()(layer_5)
layer_7 = Conv2D(filters = 10, kernel_size = 3, padding = 'same', activation = 'relu')(layer_6)
layer_8 = UpSampling2D()(layer_7)
layer_9 = Conv2D(filters = 1, kernel_size = 3, padding = 'same', activation = 'relu')(layer_8)

model = Model(inputs = inputs, outputs = layer_9, name = model_name)

model_json = model.to_json()

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )