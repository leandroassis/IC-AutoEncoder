from tensorflow.keras.callbacks import Callback


class TrainingTime (Callback):
    """
        This callback mesures the dada related to time of neural networks trainings.
    """

    def __init__(self):
        super().__init__()

        self.init_epoch_time:float = None
        self.end_epoch_time:float = None

        self.init_batch_time:float = None
        self.end_batch_time:float = None

        self.epoch_delta_times:dict = {}
        self.batch_delta_times:dict = {}

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

class TrainingStoppingCriterion (Callback):
    """
        A stop criterion to decide when stop training the neural network.
    """
    def __init__(self, criterion:function):
        super().__init__()
        self.epoch_results:list = []
        self.criterion:function = criterion

    def on_epoch_end(self, epoch, logs=None):

        self.epoch_results.append(logs)

        if self.criterion(self.epoch_results):
            self.model.stop_training()
