from abc import ABC, abstractmethod
from genericpath import isdir
from os import makedirs
import shutil
from tensorflow.python.lib.io.file_io import file_exists
from pandas import DataFrame, read_csv

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
                model_name: str, 
                dataset_name: str,
                training_idx: int,
                loss_name: str
                ) -> None:

        self.logs_dir = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}"
        
        self.model_save_pathname = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}/model"
        self.csv_pathname = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}/CsvLoger.csv"
        
        self.models_table_pathname = f'logs/models_table'
        
    def make_all_dirs (self) -> None:

        if not isdir(self.logs_dir):
            makedirs(self.logs_dir)

    def remove_last_save (self):

        if self.last_model_save and isdir(self.last_model_save):
            shutil.rmtree(self.last_model_save)

    def get_logs_size (self):
        pass

    def get_actual_training_logs_size (self):
        pass

    def _load_saved_models_file (self) -> DataFrame:
        """
        
        
        """
        if file_exists(self.models_table_pathname):

            models_saved = read_csv(self.models_table_pathname)

        else:

            models_saved = DataFrame()

        return models_saved
            
    
    def _save_pathname_in_the_list(self) -> None:
        """
        
        """
        models_saved = self._load_saved_models_file()

        models_saved["model_save_pathname"] = self.model_save_pathname

        models_saved.to_csv(self.models_table_pathname)