
from abc import ABC, abstractmethod



class TrainingDataABC (ABC):
    pass


class KerasTrainingData (TrainingDataABC):
    """
        Resposible to get all training data to be analised 
    """
    
    def __init__(self, csv_pathname, ) -> None:
        self.csv_pathname = csv_pathname
        super().__init__()

    
    def get_csv_training_history (self) -> DataFrame:
        """

        """
        dataframe = read_csv(self.csv_pathname, sep=';')
        return dataframe