import tensorflow.keras as kr
from tensorflow.keras import layers
from tensorflow.python import keras


model_file_name = "Unet.json"

#Left frist block:
inputs = kr.layers.Input(shape=(32,32,1))
Llayer1_1 = kr.layers.Conv2D(filters= 1, kernel_size=(4,4), input_shape=(32,32,1), activation='relu', padding='same') (inputs)
Llayer1_2 = kr.layers.Conv2D( filters= 2, kernel_size=(4,4), padding='same' )(Llayer1_1)
Llayer1_3 = kr.layers.Conv2D( filters= 4, kernel_size=(4,4), padding='same' )(Llayer1_2)

#Left second block:
Llayer2_1 = kr.layers.MaxPool2D()(Llayer1_3)
Llayer2_2 = kr.layers.Conv2D( filters= 8, kernel_size=(4,4), padding='same' )(Llayer2_1)
Llayer2_3 = kr.layers.Conv2D( filters= 16, kernel_size=(4,4), padding='same' )(Llayer2_2)

#Left thrid block:
Llayer3_1 = kr.layers.MaxPool2D()(Llayer2_3)
Llayer3_2 = kr.layers.Conv2D( filters= 32, kernel_size=(4,4), padding='same' )(Llayer3_1)
Llayer3_3 = kr.layers.Conv2D( filters= 64, kernel_size=(4,4), padding='same'  )(Llayer3_2)

#Left fourth block:
Llayer4_1 = kr.layers.MaxPool2D()(Llayer3_3)
Llayer4_2 = kr.layers.Conv2D( filters= 128, kernel_size=(4,4), padding='same'  )(Llayer4_1)
Llayer4_3 = kr.layers.Conv2D( filters= 256, kernel_size=(4,4), padding='same'  )(Llayer4_2)

#Right thrid block:
Rlayer3_1_2 = kr.layers.UpSampling2D()(Llayer4_3)
Rlayer3_1 = kr.layers.concatenate([Llayer3_3, Rlayer3_1_2], axis=-1)
Rlayer3_2 = kr.layers.Conv2D( filters= (64+256)//2, kernel_size=(4,4), padding='same'  )(Rlayer3_1)
Rlayer3_3 = kr.layers.Conv2D( filters=  80, kernel_size=(4,4), padding='same'  )(Rlayer3_2)

#right second block:
Rlayer2_1_2 = kr.layers.UpSampling2D()(Llayer3_3)
Rlayer2_1 = kr.layers.concatenate([Llayer2_3, Rlayer2_1_2], axis=-1)
Rlayer2_2 = kr.layers.Conv2D( filters= (16+80)//2, kernel_size=(4,4), padding='same' )(Rlayer2_1)
Rlayer2_3 = kr.layers.Conv2D( filters= 24, kernel_size=(4,4), padding='same' )(Rlayer2_2)

#right frist block:
Rlayer1_1_2 = kr.layers.UpSampling2D()(Llayer2_3)
Rlayer1_1 = kr.layers.concatenate([Llayer1_3, Rlayer1_1_2], axis=-1)
Rlayer1_2 = kr.layers.Conv2D( filters= (24+4)//2, kernel_size=(4,4), padding='same' )(Rlayer1_1)
Rlayer1_3 = kr.layers.Conv2D( filters= 7, kernel_size=(4,4), padding='same' )(Rlayer1_2)
Rlayer1_4 = kr.layers.Conv2D( filters= 4, kernel_size=(4,4), padding='same' )(Rlayer1_3)
Rlayer1_5 = kr.layers.Conv2D( filters= 2, kernel_size=(4,4), padding='same' )(Rlayer1_4)
Rlayer1_6 = kr.layers.Conv2D( filters= 1, kernel_size=(4,4), padding='same' )(Rlayer1_5)

model = kr.models.Model(inputs=inputs, outputs=Rlayer1_6, name="Unet")

model_json = model.to_json()

with open(model_file_name, "w") as json_file:
    json_file.write(model_json)