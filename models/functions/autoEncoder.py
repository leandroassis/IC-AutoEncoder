from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

import keras_tuner as kt

from modules.CustomLosses import L3SSIM
from modules.misc import ssim_metric
from modules.ImageMetrics.metrics import three_ssim, psnrb
from tensorflow.keras.optimizers import Adam

def create_AE_model(hp):

    inputs = Input(shape=(64,64,1))
    
    hp_kernel_sz = hp.Choice('kernel_size', values = [x for x in range(2, 20, 5)])
    bias = hp.Choice('bias', values = [True, False])

    l1_filters = hp.Int('l1_filters', min_value = 10, max_value = 160, step = 20)
    layer_1 = Conv2D(filters = l1_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(inputs)

    l2_filters = hp.Int('l2_filters', min_value = 10, max_value = 160, step = 20)
    layer_2 = Conv2D(filters = l2_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_1)
    layer_3 = MaxPooling2D()(layer_2)

    l4_filters = hp.Int('l3_filters', min_value = 10, max_value = 160, step = 20)
    layer_4 = Conv2D(filters = l4_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_3)

    l5_filters = hp.Int('l5_filters', min_value = 10, max_value = 160, step = 20)
    layer_5 = Conv2D(filters = l5_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_4)
    layer_6 = MaxPooling2D()(layer_5)

    l7_filters = hp.Int('l7_filters', min_value = 10, max_value = 160, step = 20)
    layer_7 = Conv2D(filters = l7_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_6)

    l8_filters = hp.Int('l8_filters', min_value = 10, max_value = 160, step = 20)
    layer_8 = Conv2D(filters = l8_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_7)
    layer_9 = MaxPooling2D()(layer_8)

    l10_filters = hp.Int('l10_filters', min_value = 10, max_value = 160, step = 20)
    layer_10 = Conv2D(filters = l10_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_9)

    l11_filters = hp.Int('l11_filters', min_value = 10, max_value = 160, step = 20)
    layer_11 = Conv2D(filters = l11_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_10)
    layer_12 = UpSampling2D()(layer_11)

    l13_filters = hp.Int('l13_filters', min_value = 10, max_value = 160, step = 20)
    layer_13 = Conv2D(filters = l13_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_12)

    l14_filters = hp.Int('l14_filters', min_value = 10, max_value = 160, step = 20)
    layer_14 = Conv2D(filters = l14_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_13)
    layer_15 = UpSampling2D()(layer_14)

    l16_filters = hp.Int('l16_filters', min_value = 10, max_value = 160, step = 20)
    layer_16 = Conv2D(filters = l16_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_15)

    l17_filters = hp.Int('l17_filters', min_value = 10, max_value = 160, step = 20)
    layer_17 = Conv2D(filters = l17_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_16)
    layer_18 = UpSampling2D()(layer_17)

    l19_filters = hp.Int('l19_filters', min_value = 10, max_value = 160, step = 20)
    layer_19 = Conv2D(filters = l19_filters, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_18)

    layer_20 = Conv2D(filters = 1, kernel_size = hp_kernel_sz, padding = 'same', activation = 'relu', use_bias=bias)(layer_19)

    model_name = "AutoEncoder-2.3-64x64"
    autoEncoder = Model(inputs = inputs, outputs = layer_20, name = model_name)


    hp_lr = hp.Choice('learning_rate', values = [0.5e-1, 1e-2, 1e-3, 1e-4])
    autoEncoder.compile(optimizer = Adam(learning_rate = hp_lr), loss = L3SSIM(), metrics=[ ssim_metric, three_ssim, psnrb ])

    return autoEncoder