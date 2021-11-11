from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any
from tensorflow.keras.models import Model, model_from_json
import json
from misc import get_neural_net_node_deep


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


    def get_all_data(self) -> dict:
        """
            Returns all data in a dict

            :Receives: None

            :Return: `dict`

            :Raise: None
        """
        return self.__dict__


    @abstractmethod
    def get_csv_data(self) -> dict:
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
            self._model = model

        if self.model == None:
            raise Exception("No model passed to te class")

        self.model_name = self.model.name

        self.total_params:int = self.model.count_params()

        self.total_layers: int = self.model.layers.__len__()

        self.total_treinable_layers: int = self._get_number_of_treinable_layers()

        self.total_regularizers: dict = self._get_number_of_regularizers()
        """A `dict` containing the total number of regularizers of each type"""

        self.regularizer_map: dict = self._get_regularizer_map()
        """A `dict` describing the distribution of regularizers"""
        
        self.regularizer_L1_map: dict = self._get_regularizer_map("L1")
        """A `dict` describing the distribution of regularizers of type L1"""

        self.regularizer_L2_map: dict = self._get_regularizer_map("L2")
        """A `dict` describing the distribution of regularizers of type L2"""

        self.regularizer_L3_map: dict = self._get_regularizer_map("L3")
        """A `dict` describing the distribution of regularizers of type L2"""

        self.parameters_distribution: int = self._get_parameters_distribution()
        """A `int` that describes how the parameters are distributed along a neural net"""

        self.initializers_count: dict = self._get_initializer_info()
        """A `dict` containing initializers count of each type."""

        self.number_segments: int = self._get_paralelism_info()
        """The number of paralel segments in the neural network"""

        dropout_count, dropout_distribution = self._get_layer_distribution_and_count("Dropout")

        self.dropout_count: int = dropout_count
        """ The number of dropout layers in the network """

        self.dropout_distribution: float = dropout_distribution
        """ The distribution of dropout layers in the network """

        batch_normalization_count, batch_normalization_distribution = self._get_layer_distribution_and_count("BatchNormalization")

        self.batch_normalization_count: int = batch_normalization_count
        """ The number of batch normalization layers in the network """

        self.batch_normalization_distribution: float = batch_normalization_distribution
        """ The distribution of batch normalization layers in the network """

        super().__init__()


    def _get_number_of_regularizers(self) -> dict:
        
        layers_config: dict = json.loads(self.model.to_json())['config']['layers']

        number_of_regularizers: dict = {
            "kernel" : 0,
            "bias" : 0,
            "activity" : 0
        }

        for layer in layers_config:
            
            try:
                if (layer['config']['kernel_regularizer'] != None):
                    number_of_regularizers["kernel"] += 1
                
                if layer['config']['bias_regularizer'] != None:
                    number_of_regularizers["bias"] += 1
                
                if layer['config']['activity_regularizer'] != None:
                    number_of_regularizers["activity"] += 1

            except KeyError:
                pass # The layer dont contain a regularizer, Ex: MaxPoll

        return number_of_regularizers


    def _get_regularizer_map(self, type: str = "") -> dict:

        layers_config: dict = json.loads(self.model.to_json())['config']['layers']

        distribution_of_regularizers: dict = {
            "kernel" : 0.0,
            "bias" : 0.0,
            "activity" : 0.0
        }

        nodes_depp = get_neural_net_node_deep(self.model)


        for layer in layers_config:

            if 'kernel_regularizer' in layer['config']:

                if (layer['config']['kernel_regularizer'] != None):
                    
                    if type == "":
                        distribution_of_regularizers["kernel"] += nodes_depp[layer['name']]
                    
                    elif layer['config']['kernel_regularizer']['class_name'] == type:
                        distribution_of_regularizers["kernel"] += nodes_depp[layer['name']]
            
            if 'bias_regularizer' in layer['config']:

                if (layer['config']['bias_regularizer'] != None):
                    
                    if type == "":
                        distribution_of_regularizers["bias"] += nodes_depp[layer['name']]
                    
                    elif layer['config']['bias_regularizer']['class_name'] == type:
                        distribution_of_regularizers["bias"] += nodes_depp[layer['name']]
            
            
            if 'activity_regularizer' in layer['config']:

                if (layer['config']['activity_regularizer'] != None):
                    
                    if type == "":
                        distribution_of_regularizers["activity"] += nodes_depp[layer['name']]
            
                    elif layer['config']['activity_regularizer']['class_name'] == type:
                        distribution_of_regularizers["activity"] += nodes_depp[layer['name']]
            
        max_depp = (max(nodes_depp.values()))

        if self.total_regularizers['activity'] > 0:
            distribution_of_regularizers['activity'] = distribution_of_regularizers['activity']/( (max_depp)*(self.total_regularizers['activity']))
        
        if self.total_regularizers['bias'] > 0: 
            distribution_of_regularizers['bias'] = distribution_of_regularizers['bias']/( (max_depp)*(self.total_regularizers['bias']))

        if self.total_regularizers['kernel'] > 0:
            distribution_of_regularizers['kernel'] = distribution_of_regularizers['kernel']/( (max_depp)*(self.total_regularizers['kernel']) )

        return distribution_of_regularizers


    def _get_paralelism_info(self):
        """
            Get the number of paralel sequences of neurons, with the number of inbound nodes of the layers.
        """

        info: dict = {
            'number' : 1
        }

        layers_config: dict = json.loads(self.model.to_json())['config']['layers']
        
        for layer in layers_config:

            if layer['class_name'] == "Concatenate":
                info['number'] += layer['inbound_nodes'][0].__len__() - 1

            
        return info['number']


    def _get_parameters_distribution(self) -> int:

        nodes_depp = get_neural_net_node_deep(self.model)

        distribution: float = 0.0
        
        for layer in self.model.layers:

            distribution += layer.count_params()*nodes_depp[layer.name]
            
        distribution = distribution/(self.total_params*self.total_layers)
            
        return distribution



    def _get_initializer_info(self) -> dict:
        """
            Get the intitializer type count for bias and kernel
        """

        count_of_initializers: dict = {
            "kernel" : {},
            "bias" : {},
            }

        kernel_initializer_type: dict = {
        }

        bias_initializer_type: dict = {
        }

        layers_config: dict = json.loads(self.model.to_json())['config']['layers']

        for layer in layers_config:

            if 'kernel_initializer' in layer['config']:
                kernel_initializer = layer['config']['kernel_initializer']['class_name']

                if kernel_initializer in kernel_initializer_type:
                    kernel_initializer_type[kernel_initializer] += 1
                else:
                    kernel_initializer_type[kernel_initializer] = 1

            if 'bias_initializer' in layer['config']:
                bias_initializer = layer['config']['bias_initializer']['class_name']

                if bias_initializer in bias_initializer_type:
                    bias_initializer_type[bias_initializer] += 1
                else:
                    bias_initializer_type[bias_initializer] = 1
        
        count_of_initializers['kernel'] = kernel_initializer_type
        count_of_initializers['bias'] = bias_initializer_type

        return count_of_initializers


    def _get_number_of_treinable_layers(self) -> int:

        total_layers = 0

        for layer in self.model.layers:

            if layer.count_params() > 0:

                total_layers += 1
                
        return total_layers


    def get_csv_data(self) -> dict:
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

    def _get_layer_distribution_and_count (self, layer_class_name) -> tuple:

        layers_config: dict = json.loads(self.model.to_json())['config']['layers']

        nodes_depp: dict = get_neural_net_node_deep(self.model)

        count: int = 0

        distribution: float = 0

        for layer in layers_config:

            if layer['class_name'] == layer_class_name:
                count += 1
                distribution += nodes_depp[layer['name']]

        if count:
            distribution = distribution/(count*max(nodes_depp.values()))

        return count, distribution

    
    def get_model(self) -> Model:
        """
            Model to be used in trainings, predicts...
        """
        return self.model



