from abc import ABC, abstractmethod

class DataSetABC (ABC):
    """
        This class 
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def get_metadata(self) -> dict:
        """
            Method that returns all metadata relative to the states off the data set
        """
        pass