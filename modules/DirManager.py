from abc import ABC, abstractmethod
from genericpath import isdir
from os import makedirs
import shutil
from tensorflow.python.lib.io.file_io import file_exists


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

    def __init__(self, 
                model_name: str, 
                dataset_name: str,
                training_idx: int,
                loss_name: str
                ) -> None:

        self.logs_dir = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}"
        
        self.model_save_pathname = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}/model"
        self.csv_pathname = f"logs/{dataset_name}/{loss_name}/{model_name}/{training_idx}/CsvLoger.csv"
        
        self.last_model_save_pathname = f'logs/last_model_saved'
        self.models_saved_pathname = f'logs/models_saved_pathname'
        self.last_model_save = self._load_last_model_pathname()
        
    def make_all_dirs(self) -> None:

        if not isdir(self.logs_dir):
            makedirs(self.logs_dir)

    def remove_last_save(self):

        if self.last_model_save and isdir(self.last_model_save):
            shutil.rmtree(self.last_model_save)
        
    def _load_last_model_pathname(self) -> str:

        if file_exists(self.last_model_save_pathname):
            with open(self.last_model_save_pathname, 'r') as file:
                path = file.read()
                file.close()

                return path

    def save_actual_model_pathname_as_last(self) -> None:
        with open(f'logs/last_model_saved', 'w') as file:
            file.write(self.model_save_pathname)


    def _save_pathname_in_the_list() -> None:
        pass


    def get_logs_size(self):
        pass

    def get_actual_training_logs_size(self):
        pass