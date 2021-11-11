from abc import ABC, abstractmethod
import tensorflow as tf




class TensorBoardWriterABC (ABC):
    """
        This class 
    """

    @abstractmethod
    def __init__(self) -> None:
        pass



class TensorBoardWriter (TensorBoardWriterABC):
    """

    """

    def __init__(self, file_path) -> None:
        self.file_path = file_path + "/TensoboardData"


    def write_images (self, images: list, name:str) -> None:
        """

        """

        writer = tf.summary.create_file_writer(self.file_path)
        with writer.as_default():
            tf.summary.image(name, images, step=0, max_outputs= 4)
            writer.flush()

    def write_scalars (self, scalars) -> None:
        """
            
        """
        pass


    