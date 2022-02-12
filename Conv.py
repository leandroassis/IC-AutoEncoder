from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "Conv-12.0-64x64"

inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 3, kernel_size = 3, padding = 'same', activation = 'relu')(inputs)
layer_2 = Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu')(layer_1)
layer_3 = Conv2D(filters = 24, kernel_size = 3, padding = 'same', activation = 'relu')(layer_2)
layer_4 = Conv2D(filters = 40, kernel_size = 3, padding = 'same', activation = 'relu')(layer_3)
layer_5 = Conv2D(filters = 44, kernel_size = 3, padding = 'same', activation = 'relu')(layer_4)
layer_6 = Conv2D(filters = 50, kernel_size = 3, padding = 'same', activation = 'relu')(layer_5)
layer_7 = Conv2D(filters = 44, kernel_size = 3, padding = 'same', activation = 'relu')(layer_6)
layer_8 = Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')(layer_7)
layer_9 = Conv2D(filters = 24, kernel_size = 3, padding = 'same', activation = 'relu')(layer_8)
layer_10 = Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu')(layer_9)
layer_11 = Conv2D(filters = 3, kernel_size = 3, padding = 'same', activation = 'relu')(layer_10)
layer_12 = Conv2D(filters = 1, kernel_size = 3, padding = 'same', activation = 'relu')(layer_11)

model = Model(inputs = inputs, outputs = layer_12, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )