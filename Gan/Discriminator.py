from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers.core import Dense, Dropout

from os import environ

environ["CUDA_VISIBLE_DEVICES"]="1"
model_name = "Discriminator-AutoEncoder-1.0-64x64"


inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 10, kernel_size = 3, padding = 'same', activation = 'relu')(inputs)
layer_2 = MaxPooling2D()(layer_1)
layer_3 = Conv2D(filters = 20, kernel_size = 3, padding = 'same', activation = 'relu')(layer_2)
layer_4 = BatchNormalization()(layer_3)
layer_5 = MaxPooling2D()(layer_4)
layer_6 = Conv2D(filters = 30, kernel_size = 3, padding = 'same', activation = 'relu')(layer_5)
layer_7 = MaxPooling2D()(layer_6)
layer_8 = Conv2D(filters = 60, kernel_size = 3, padding = 'same', activation = 'relu')(layer_7)
layer_9 = Conv2D(filters = 70, kernel_size = 3, padding = 'same', activation = 'relu')(layer_8)
layer_10 = MaxPooling2D()(layer_9)
layer_11 = Conv2D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu')(layer_10)
layer_12 = Flatten()(layer_11)
layer_13 = Dense(400, activation='relu')(layer_12)
layer_14 = Dropout(0.3)(layer_13)
layer_15 = Dense(50, activation='relu')(layer_14)
layer_16 = Dense(1, activation='sigmoid')(layer_15)

model = Model(inputs = inputs, outputs = layer_16, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested = True )