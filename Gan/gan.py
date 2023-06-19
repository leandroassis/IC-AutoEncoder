from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add , Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "Generator-Dense-1.0"

inputs = Input(shape=(100))
layer_1 = Dense(units = 50, activation = 'relu')(inputs)
layer_2 = Dense(units = 80, activation = 'relu')(layer_1)
layer_3 = Dense(units = 100, activation = 'relu')(layer_2)
layer_4 = Dense(units = 120, activation = 'relu')(layer_3)
layer_5 = Dense(units = 150, activation = 'relu')(layer_4)
layer_6 = Dense(units = 100, activation = None)(layer_5)

model = Model(inputs = inputs, outputs = layer_6, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )

print(model.count_params())