from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add , Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "ResidualAutoEncoder-0.1-64x64"

inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 1, kernel_size = 4, padding = 'same', activation = 'relu')(inputs)
layer_2 = Add()([layer_1, inputs])
layer_3 = Conv2D(filters = 20, kernel_size = 4, padding = 'same', activation = 'relu')(layer_2)
layer_4 = Conv2D(filters = 50, kernel_size = 4, padding = 'same', activation = 'relu')(layer_3)
layer_5 = Conv2D(filters = 20, kernel_size = 4, padding = 'same', activation = 'relu')(layer_4)
layer_6 = Add()([layer_3, layer_5])
layer_7 = Conv2D(filters = 1, kernel_size = 4, padding = 'same', activation = 'relu')(layer_6)

model = Model(inputs = inputs, outputs = layer_7, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/GAN" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )

print(model.count_params())