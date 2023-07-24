from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

model_name = "AutoEncoder-2.3-64x64"

inputs = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 20, kernel_size = 4, padding = 'same', activation = 'relu')(inputs)
layer_2 = Conv2D(filters = 40, kernel_size = 4, padding = 'same', activation = 'relu')(layer_1)
layer_3 = MaxPooling2D()(layer_2)
layer_4 = Conv2D(filters = 60, kernel_size = 3, padding = 'same', activation = 'relu')(layer_3)
layer_5 = Conv2D(filters = 70, kernel_size = 3, padding = 'same', activation = 'relu')(layer_4)
layer_6 = MaxPooling2D()(layer_5)
layer_7 = Conv2D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu')(layer_6)
layer_8 = Conv2D(filters = 120, kernel_size = 3, padding = 'same', activation = 'relu')(layer_7)
layer_9 = MaxPooling2D()(layer_8)
layer_10 = Conv2D(filters = 120, kernel_size = 3, padding = 'same', activation = 'relu')(layer_9)
layer_11 = Conv2D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu')(layer_10)
layer_12 = UpSampling2D()(layer_11)
layer_13 = Conv2D(filters = 80, kernel_size = 3, padding = 'same', activation = 'relu')(layer_12)
layer_14 = Conv2D(filters = 60, kernel_size = 3, padding = 'same', activation = 'relu')(layer_13)
layer_15 = UpSampling2D()(layer_14)
layer_16 = Conv2D(filters = 50, kernel_size = 3, padding = 'same', activation = 'relu')(layer_15)
layer_17 = Conv2D(filters = 25, kernel_size = 3, padding = 'same', activation = 'relu')(layer_16)
layer_18 = UpSampling2D()(layer_17)
layer_19 = Conv2D(filters = 5, kernel_size = 4, padding = 'same', activation = 'relu')(layer_18)
layer_20 = Conv2D(filters = 1, kernel_size = 4, padding = 'same', activation = 'relu')(layer_19)


autoEncoder = Model(inputs = inputs, outputs = layer_20, name = model_name)

model_json = autoEncoder.to_json(indent = 4)

with open("models/arch/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)

plot_model(model=autoEncoder, to_file="models/arch/photos/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )

print(autoEncoder.count_params())