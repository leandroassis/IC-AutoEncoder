# trainer com mlflow pra lgoar + cross validation + treinar até não ter melhoria

from modules.DataMod import DataSet
from numpy import ndarray

class NetTrainer():

    def __init__(self, model = None,
                 train_dataset : DataSet = None,
                 validation_dataset : ndarray = None,
                 validation_percentage : float = 0.15,
                 ):
        
        self.model = None
        self.dataset = None

        self.num_epochs = 0
        self.batch_size = 0
        self.loss = None

        self.set_model(model)


    def set_model(self, model) -> None:
        if self.model is not None:
            print("Warning: model is not None. This will overwrite the current model.")
        
        self.model = model

    def set_dataset(self, dataset : DataSet) -> None:
        if self.dataset is not None:
            print("Warning: dataset is not None. This will overwrite the current dataset.")

        self.dataset : DataSet = dataset

    def start_training(self) -> None:
        pass


    def log_training(self) -> None:
        pass