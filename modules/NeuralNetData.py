from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any
from tensorflow.keras.models import Model, model_from_json
from KerasModelInfo import *



@dataclass
class NeuralNetDataABC (ABC):
    """
        This class contains all neural netword data. It also returns all interesting data to be analised.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
            Initialize all data of the neural network.
        """
        super().__init__()


    @abstractmethod
    def get_model_csv_data(self) -> dict:
        """
            Get all data to save in the table of training data.

            :Receives: None

            :Return: `dict`

            :Raise: None

        """
        pass

    
    

class KerasNeuralNetData(NeuralNetDataABC):
    """
        NeuralNet data for keras Models
    """

    def __init__(self, model: Model = None, 
                       model_name: str = None,
                       load_model: bool = False,
                       models_dir : str = "nNet_models/"
                       ) -> None:
    
        if load_model:
            
            with open(models_dir + model_name, 'r') as json_file:
                architecture = json_file.read()
                self.model = model_from_json(architecture)
                json_file.close()

        else:
            self.model = model

        if self.model == None:
            raise Exception("No model passed to the class")

        self.model_name = self.model.name

        self.total_params:int = self.model.count_params()

        self.total_layers: int = self.model.layers.__len__()

        self.total_treinable_layers: int = get_number_of_layers_with_params(self.model)

        self.total_regularizers: dict = get_number_of_regularizers(self.model)
        """A `dict` containing the total number of regularizers of each type"""

        self.regularizer_map: dict = get_regularizer_distribution(self.model)
        """A `dict` describing the distribution of regularizers"""
        
        self.regularizer_L1_map: dict = get_regularizer_distribution(self.model, "L1")
        """A `dict` describing the distribution of regularizers of type L1"""

        self.regularizer_L2_map: dict = get_regularizer_distribution(self.model, "L2")
        """A `dict` describing the distribution of regularizers of type L2"""

        self.parameters_distribution: int = get_parameters_distribution(self.model)
        """A `int` that describes how the parameters are distributed along a neural net"""

        self.initializers_count: dict = get_initializer_info(self.model)
        """A `dict` containing initializers count of each type."""

        self.number_segments: int = get_paralelism_info(self.model)
        """The number of paralel segments in the neural network"""

        dropout_count, dropout_distribution = get_layer_distribution_and_count(self.model, "Dropout")

        self.dropout_count: int = dropout_count
        """ The number of dropout layers in the network """

        self.dropout_distribution: float = dropout_distribution
        """ The distribution of dropout layers in the network """

        batch_normalization_count, batch_normalization_distribution = get_layer_distribution_and_count(self.model ,"BatchNormalization")

        self.batch_normalization_count: int = batch_normalization_count
        """ The number of batch normalization layers in the network """

        self.batch_normalization_distribution: float = batch_normalization_distribution
        """ The distribution of batch normalization layers in the network """

        super().__init__()


    def get_model_csv_data(self) -> dict:
        """
            Get all data to save in the table of training data.

            :receives: None
            :return: `dict`
            :raise: None
        """

        data = {
            "model_name" : self.model_name,
            "total_params" : self.total_params,
            "params_distribution" : self.parameters_distribution,
            "total_layers" : self.total_layers,
            "total_treinable_layers" : self.total_treinable_layers,
            "total_paths" : self.number_segments,
            "total_kernel_regularizers" : self.total_regularizers['kernel'],
            "total_bias_regularizers" : self.total_regularizers['bias'],
            "total_activity_regularizers" : self.total_regularizers['activity'],
            "kernel_regularizers_distribution" : self.regularizer_map['kernel'],
            "bias_regularizers_distribution" : self.regularizer_map['bias'],
            "activity_regularizers_distribution" : self.regularizer_map['activity'],
            "kernel_regularizers_L1_distribution" : self.regularizer_L1_map['kernel'],
            "bias_regularizers_L1_distribution" : self.regularizer_L1_map['bias'],
            "activity_regularizers_L1_distribution" : self.regularizer_L1_map['activity'],
            "kernel_regularizers_L2_distribution" : self.regularizer_L2_map['kernel'],
            "bias_regularizers_L2_distribution" : self.regularizer_L2_map['bias'],
            "activity_regularizers_L2_distribution" : self.regularizer_L2_map['activity'],
            "initializer_count" : self.initializers_count,
            "dropout_count" : self.dropout_count,
            "dropout_distribution" : self.dropout_distribution,
            "batch_normalization_count" : self.batch_normalization_count,
            "batch_normalization_distribution" : self.batch_normalization_distribution
        }

        return data

    
    def get_model(self) -> Model:
        """
            Model to be used in trainings, predicts...
        """
        return self.model



