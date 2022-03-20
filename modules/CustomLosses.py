from tensorflow.keras.losses import Loss, Reduction, MeanAbsoluteError, binary_crossentropy
from misc import get_model
import tensorflow as tf
from tensorflow._api.v2.image import ssim

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


class AdversarialLoss(Loss):

    def __init__(self, training_idx: int = None, model_name: str = None, custom_objects: dict = None, reduction=Reduction.AUTO, name: str = 'AdversarialLoss'):
        
        super().__init__(reduction=reduction, name=name)

        if training_idx == None and model_name == None:
            raise Exception("No model has bem passed, set a model name or training_idx")

        if model_name:
            self.adversarial_model = get_model(model_name = model_name)

        if training_idx != None:
            self.adversarial_model = get_model(training_idx = training_idx)
    
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
