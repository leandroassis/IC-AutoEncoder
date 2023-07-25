from abc import ABC, abstractmethod
from genericpath import isdir
from os import makedirs

class DirManagerABC (ABC):
    """
        This class 
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def make_all_dirs(self) -> None:
        pass
        


class KerasDirManager (DirManagerABC):

    def __init__ (self,
                logs_dir_name:str,
                models_csv_name:str,
                trainings_csv_name:str,
                model_name: str, 
                dataset_name: str,
                training_idx: int,
                ) -> None:

        self.base_dir = f"{logs_dir_name}/{dataset_name}/{model_name}/{training_idx}"
        self.model_save_best_training_pathname = self.base_dir + "/model_best_training_checkpoint"
        self.model_save_best_validation_pathname = self.base_dir + "/model_best_validation_checkpoint"
        
        self.metric_means_pathname = f"{logs_dir_name}/{dataset_name}/{model_name}/{training_idx}/metrics_epoch_means.csv"

        self.models_table_pathname = f"{logs_dir_name}/{models_csv_name}.csv"
        self.trainings_table_pathname = f"{logs_dir_name}/{trainings_csv_name}.csv"

        self.make_all_dirs()
        
    def make_all_dirs (self) -> None:

        if not isdir(self.base_dir):
            makedirs(self.base_dir)


