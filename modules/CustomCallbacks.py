from tensorflow.keras.callbacks import Callback, EarlyStopping
from time import time

class TrainingTime (Callback):
    """
        This callback mesures the dada related to time of neural networks trainings.
    """

    def __init__(self):
        super().__init__()

        self.init_epoch_time:float = None
        self.end_epoch_time:float = None
        self.train_time:float = None

        self.init_train_time:float = None
        self.end_train_time:float = None

        self.epoch_delta_times:list = []

    def on_train_begin(self, logs=None):
        self.init_train_time = time()

    def on_train_end(self, logs=None):
        self.end_train_time = time()
        self.train_time = self.init_train_time - self.end_train_time
    
    def on_epoch_begin(self, epoch, logs=None):
        self.init_epoch_time = time()

    def on_epoch_end(self, epoch, logs=None):
        self.end_epoch_time = time()
        self.epoch_delta_times.append(self.init_epoch_time - self.end_epoch_time)


class TrainingStoppingCriterion (Callback):
    """
        A stop criterion to decide when stop training the neural network.
    """
    def __init__(self, criterion:function, metric_name:str = "loss", function_kwargs:dict = {}):
        super().__init__()
        self.epoch_results:list = []
        self.metric_name = metric_name
        self.function_kwargs = function_kwargs
        self.stop_criterion_is_true:function = criterion

    def on_epoch_end(self, epoch, logs=None):
        
        if logs:
            self.epoch_results = logs.get(self.metric_name)

        if self.stop_criterion_is_true (self.epoch_results, self.metric_name, **self.function_kwargs):
            self.model.stop_training = True

            