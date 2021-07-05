import tensorflow.keras as kr
from tensorflow.keras import layers
from tensorflow.python import keras


model_file_name = "Unet.json"

#Left frist block:
inputs = kr.layers.Input(shape=(32,32,1))
LeftLayer1_1 = kr.layers.Conv2D(filters= 4, kernel_size=8, input_shape=(32,32,1), activation='relu', padding='same') (inputs)
LeftLayer1_2 = kr.layers.Conv2D( filters= 5, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_1)
LeftLayer1_3 = kr.layers.Conv2D( filters= 6, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_2)
LeftLayer1_4 = kr.layers.Conv2D( filters= 7, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_3)
LeftLayer1_5 = kr.layers.Conv2D( filters= 8, kernel_size=8, activation='relu', padding='same' )(LeftLayer1_4)

#Left second block:
LeftLayer2_1 = kr.layers.MaxPool2D()(LeftLayer1_5)
LeftLayer2_2 = kr.layers.Conv2D( filters= 10, kernel_size=6, activation='relu' , padding='same' )(LeftLayer2_1)
LeftLayer2_3 = kr.layers.Conv2D( filters= 12, kernel_size=6, activation='relu' , padding='same' )(LeftLayer2_2)
LeftLayer2_4 = kr.layers.Conv2D( filters= 14, kernel_size=6, activation='relu' , padding='same' )(LeftLayer2_3)
LeftLayer2_5 = kr.layers.Conv2D( filters= 16, kernel_size=6, activation='relu' , padding='same' )(LeftLayer2_4)
LeftLayer2_6 = kr.layers.Conv2D( filters= 18, kernel_size=6, activation='relu' , padding='same' )(LeftLayer2_5)

#Left thrid block:
LeftLayer3_1 = kr.layers.MaxPool2D()(LeftLayer2_6)
LeftLayer3_2 = kr.layers.Conv2D( filters= 32, kernel_size=4, activation='relu' , padding='same' )(LeftLayer3_1)
LeftLayer3_3 = kr.layers.Conv2D( filters= 40, kernel_size=4, activation='relu' , padding='same'  )(LeftLayer3_2)
LeftLayer3_4 = kr.layers.Conv2D( filters= 46, kernel_size=4, activation='relu' , padding='same'  )(LeftLayer3_3)
LeftLayer3_5 = kr.layers.Conv2D( filters= 54, kernel_size=4, activation='relu' , padding='same'  )(LeftLayer3_4)
LeftLayer3_6 = kr.layers.Conv2D( filters= 64, kernel_size=4, activation='relu' , padding='same'  )(LeftLayer3_5)

#Left fourth block:
LeftLayer4_1 = kr.layers.MaxPool2D()(LeftLayer3_6)
LeftLayer4_2 = kr.layers.Conv2D( filters= 128, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_1)
LeftLayer4_3 = kr.layers.Conv2D( filters= 142, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_2)
LeftLayer4_4 = kr.layers.Conv2D( filters= 154, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_3)
LeftLayer4_5 = kr.layers.Conv2D( filters= 166, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_4)
LeftLayer4_6 = kr.layers.Conv2D( filters= 184, kernel_size=2, activation='relu' , padding='same'  )(LeftLayer4_5)

#Right thrid block:
RightLayer3_1_2 = kr.layers.UpSampling2D()(LeftLayer4_6)
RightLayer3_1 = kr.layers.concatenate([LeftLayer3_6, RightLayer3_1_2], axis=-1)
RightLayer3_2 = kr.layers.Conv2D( filters=  270, kernel_size=4, activation='relu' , padding='same'  )(RightLayer3_1)
RightLayer3_3 = kr.layers.Conv2D( filters=  210, kernel_size=4, activation='relu' , padding='same'  )(RightLayer3_2)
RightLayer3_4 = kr.layers.Conv2D( filters=  180, kernel_size=4, activation='relu' , padding='same'  )(RightLayer3_3)
RightLayer3_5 = kr.layers.Conv2D( filters=  162, kernel_size=4, activation='relu' , padding='same'  )(RightLayer3_4)
RightLayer3_6 = kr.layers.Conv2D( filters=  128, kernel_size=4, activation='relu' , padding='same'  )(RightLayer3_5)

#right second block:
RightLayer2_1_2 = kr.layers.UpSampling2D()(RightLayer3_6)
RightLayer2_1 = kr.layers.concatenate([LeftLayer2_6, RightLayer2_1_2], axis=-1)
RightLayer2_2 = kr.layers.Conv2D( filters= 64, kernel_size=6, activation='relu' , padding='same' )(RightLayer2_1)
RightLayer2_3 = kr.layers.Conv2D( filters= 48, kernel_size=6, activation='relu' , padding='same' )(RightLayer2_2)
RightLayer2_4 = kr.layers.Conv2D( filters= 36, kernel_size=6, activation='relu' , padding='same' )(RightLayer2_3)
RightLayer2_5 = kr.layers.Conv2D( filters= 24, kernel_size=6, activation='relu' , padding='same' )(RightLayer2_4)
RightLayer2_6 = kr.layers.Conv2D( filters= 16, kernel_size=6, activation='relu' , padding='same' )(RightLayer2_5)

#right frist block:
RightLayer1_1_2 = kr.layers.UpSampling2D()(RightLayer2_6)
RightLayer1_1 = kr.layers.concatenate([LeftLayer1_5, RightLayer1_1_2], axis=-1)
RightLayer1_2 = kr.layers.Conv2D( filters= 12, kernel_size=8 , activation='relu' , padding='same' )(RightLayer1_1)
RightLayer1_3 = kr.layers.Conv2D( filters= 8, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_2)
RightLayer1_4 = kr.layers.Conv2D( filters= 6, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_3)
RightLayer1_5 = kr.layers.Conv2D( filters= 4, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_4)
RightLayer1_6 = kr.layers.Conv2D( filters= 2, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_5)
RightLayer1_7 = kr.layers.Conv2D( filters= 1, kernel_size=8, activation='relu' , padding='same' )(RightLayer1_6)

model = kr.models.Model(inputs=inputs, outputs=RightLayer1_7, name="Unet")

model_json = model.to_json()

model.summary()

with open(model_file_name, "w") as json_file:
    json_file.write(model_json)