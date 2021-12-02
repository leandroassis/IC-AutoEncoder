from abc import ABC, abstractmethod
from os import write
from pandas.core.frame import DataFrame
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops import summary_ops_v2
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
        writer.close()


    def write_net_graph (self, model: Model):

        writer = tf.summary.create_file_writer(self.file_path)
        with writer.as_default():
            summary_ops_v2.keras_model('keras', model, step=0)
        writer.close()


    def write_scalars (self, scalars: DataFrame) -> None:
        """
            scalars:
                A Dataframe with epoch loss and metrics
        """

        writer = tf.summary.create_file_writer(self.file_path)
        with writer.as_default():
            tf.summary.scalar()
        writer.close()
        


    