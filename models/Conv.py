from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "Conv-1.3-64x64"

inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 1, kernel_size = 5, padding = 'same', activation = 'relu')(inputs)

model = Model(inputs = inputs, outputs = layer_1, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )