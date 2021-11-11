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

    def __init__(self, 
                model_name: str, 
                dataset_name: str,
                training_idx: int, 
                ) -> None:

        self.logs_dir = f"logs/{dataset_name}/{model_name}/{training_idx}"
        self.model_save_pathname = f"logs/{dataset_name}/{model_name}/{training_idx}/model"
        self.csv_pathname = f"logs/{dataset_name}/{model_name}/{training_idx}/CsvLoger.csv"
        
    def make_all_dirs(self) -> None:

        if not isdir(self.logs_dir):
            makedirs(self.logs_dir)
        