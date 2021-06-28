import tensorflow.keras as kr
from tensorflow.keras import layers
from tensorflow.python import keras


model_file_name = "Unet.json"

#Left frist block:
inputs = kr.layers.Input(shape=(32,32,1))
LeftLayer1_1 = kr.layers.Conv2D(filters= 2, kernel_size=8, input_shape=(32,32,1), activation='relu', padding='same') (inputs)
LeftLayer1_2 = kr.layers.Conv2D( filters= 3, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_1)
LeftLayer1_3 = kr.layers.Conv2D( filters= 4, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_2)

#Left second block:
LeftLayer2_1 = kr.layers.MaxPool2D()(LeftLayer1_3)
LeftLayer2_2 = kr.layers.Conv2D( filters= 8, kernel_size=4, activation='relu' , padding='same' )(LeftLayer2_1)
LeftLayer2_3 = kr.layers.Conv2D( filters= 16, kernel_size=4, activation='relu' , padding='same' )(LeftLayer2_2)

#Left thrid block:
LeftLayer3_1 = kr.layers.MaxPool2D()(LeftLayer2_3)
LeftLayer3_2 = kr.layers.Conv2D( filters= 32, kernel_size=3, activation='relu' , padding='same' )(LeftLayer3_1)
LeftLayer3_3 = kr.layers.Conv2D( filters= 64, kernel_size=3, activation='relu' , padding='same'  )(LeftLayer3_2)

#Left fourth block:
LeftLayer4_1 = kr.layers.MaxPool2D()(LeftLayer3_3)
LeftLayer4_2 = kr.layers.Conv2D( filters= 128, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_1)
LeftLayer4_3 = kr.layers.Conv2D( filters= 256, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_2)

#Right thrid block:
RightLayer3_1_2 = kr.layers.UpSampling2D()(LeftLayer4_3)
RightLayer3_1 = kr.layers.concatenate([LeftLayer3_3, RightLayer3_1_2], axis=-1)
RightLayer3_2 = kr.layers.Conv2D( filters= (64+256)//2, kernel_size=3, activation='relu' , padding='same'  )(RightLayer3_1)
RightLayer3_3 = kr.layers.Conv2D( filters=  80, kernel_size=3, activation='relu' , padding='same'  )(RightLayer3_2)

#right second block:
RightLayer2_1_2 = kr.layers.UpSampling2D()(RightLayer3_3)
RightLayer2_1 = kr.layers.concatenate([LeftLayer2_3, RightLayer2_1_2], axis=-1)
RightLayer2_2 = kr.layers.Conv2D( filters= (16+80)//2, kernel_size=4, activation='relu' , padding='same' )(RightLayer2_1)
RightLayer2_3 = kr.layers.Conv2D( filters= 24, kernel_size=4, activation='relu' , padding='same' )(RightLayer2_2)

#right frist block:
RightLayer1_1_2 = kr.layers.UpSampling2D()(RightLayer2_3)
RightLayer1_1 = kr.layers.concatenate([LeftLayer1_3, RightLayer1_1_2], axis=-1)
RightLayer1_2 = kr.layers.Conv2D( filters= (24+4)//2, kernel_size=(4,4), activation='relu' , padding='same' )(RightLayer1_1)
RightLayer1_3 = kr.layers.Conv2D( filters= 7, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_2)
RightLayer1_4 = kr.layers.Conv2D( filters= 4, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_3)
RightLayer1_5 = kr.layers.Conv2D( filters= 2, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_4)
RightLayer1_6 = kr.layers.Conv2D( filters= 1, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_5)

model = kr.models.Model(inputs=inputs, outputs=RightLayer1_6, name="Unet")

model_json = model.to_json()

model.summary()

with open(model_file_name, "w") as json_file:
    json_file.write(model_json)