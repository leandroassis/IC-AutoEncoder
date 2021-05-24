import tensorflow.keras as kr
from tensorflow.keras import layers
from tensorflow.python import keras


#Left frist block:
inputs = kr.layers.Conv2D()
Llayer1_2 = kr.layers.Conv2D()(inputs)
Llayer1_3 = kr.layers.Conv2D()(Llayer1_2)

#Left second block:
Llayer2_1 = kr.layers.MaxPool2D()(Llayer1_3)
Llayer2_2 = kr.layers.Conv2D()(Llayer2_1)
Llayer2_3 = kr.layers.Conv2D()(Llayer2_2)

#Left thrid block:
Llayer3_1 = kr.layers.MaxPool2D()(Llayer2_3)
Llayer3_2 = kr.layers.Conv2D()(Llayer3_1)
Llayer3_3 = kr.layers.Conv2D()(Llayer3_2)

#Right thrid block:
Rlayer3_1 = kr.layers.concatenate([Llayer3_3, ])
