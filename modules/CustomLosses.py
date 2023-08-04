from tensorflow.keras.losses import Loss, Reduction, MeanAbsoluteError, binary_crossentropy
from modules.misc import get_model
import tensorflow as tf
from tensorflow._api.v2.image import ssim
from ImageMetrics.metrics import three_ssim, psnrb
from tensorflow import Tensor

class LSSIM (Loss):

    def __init__(self, max_val = 255, filter_size=9, filter_sigma=1.5, k1=0.01, k2=0.03, name = "LSSIM", reduction = Reduction.AUTO) -> None:
        
        super(LSSIM, self).__init__(name = name, reduction = reduction)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

    
    def call (self,y_true,y_pred):
        return 1-ssim(y_true, y_pred, max_val = self.max_val,
                      filter_size = self.filter_size,
                      filter_sigma = self.filter_sigma,
                      k1 = self.k1,
                      k2 = self.k2)

class L3SSIM (Loss):

    def __init__(self, 
                max_val: float = 255,
                weight_for_edges: int = 2,
                weight_for_texture: int = 1,
                weight_for_smooth: int = 1,
                threshold_for_edges: float = 0.12,
                threshold_for_textures: float = 0.06,
                filter_size: int = 11,
                filter_sigma: float = 1.5,
                k1: float = 0.01,
                k2: float = 0.03,
                keep_padding: bool = True,
                name = "L3SSIM",
                reduction = Reduction.AUTO) -> None:
        
        super(L3SSIM, self).__init__(name = name, reduction = reduction)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.weight_for_edges = weight_for_edges
        self.weight_for_texture = weight_for_texture
        self.weight_for_smooth = weight_for_smooth
        self.threshold_for_edges = threshold_for_edges
        self.threshold_for_textures = threshold_for_textures
        self.keep_padding = keep_padding

    
    def call (self,y_true,y_pred):

        return 1-three_ssim(original_images=y_true, degraded_images = y_pred, max_val = self.max_val,
                      filter_size = self.filter_size,
                      filter_sigma = self.filter_sigma,
                      k1 = self.k1,
                      k2 = self.k2,
                      weight_for_edges = self.weight_for_edges,
                      weight_for_smooth = self.weight_for_smooth,
                      weight_for_texture = self.weight_for_texture,
                      threshold_for_edges = self.threshold_for_edges,
                      threshold_for_textures = self.threshold_for_textures,
                      keep_padding = self.keep_padding)


class LPSNRB (Loss):

    def __init__(self, reduction=Reduction.AUTO, name="LPSNRB"):
        super(LPSNRB, self).__init__(reduction, name)

    def call(self, y_true, y_pred):
        return - psnrb (degraded_imgs=y_pred, target_imgs=y_true)


class AdversarialLoss(Loss):

    def __init__(self, training_idx: int = None, model = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None and model == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)

        if model != None:
            self.adversarial_model = model
    
    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        return binary_crossentropy(y_pred = self.adversarial_model(y_pred), y_true = tf.ones(shape= (y_pred.shape[0], 1)))

    
class L1AdversarialLoss(Loss):

    def __init__(self, w1 = 1, w2 = 1, training_idx: int = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'L1_AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)

        self.w1 = w1
        self.w2 = w2
    
    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        return self.w1*binary_crossentropy(y_pred = self.adversarial_model(y_pred), 
                                    y_true = tf.ones(shape= (y_pred.shape[0], 1))) + tf.reduce_mean(self.w2*MeanAbsoluteError().call(y_true, y_pred), axis = (1,2))
