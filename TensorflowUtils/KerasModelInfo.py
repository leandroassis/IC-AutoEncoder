"""
    Model Info
    ==========


"""
import json
from tensorflow.python.keras.engine.training import Model

class function:
    pass


def max_neural_net_deep(model: Model) -> dict:
    """
        This function will define the deep of a neuron with the longest path from the initial node in graph network

        Receives: 
            kr.models.Model
        
        Returns: 
            `dict` containing {'name':'deep', ...}
        
        Raises: 
            Nothing
    """
    
    layers_config: dict = json.loads(model.to_json())['config']['layers']

    layers_deep: dict = {}

    for layer in layers_config:

        if layer['inbound_nodes'].__len__() == 0:
            layers_deep[layer['name']] = 0

        elif layer['class_name'] == "Concatenate":

            list_of_preveous_layers = [name[0] for name in layer['inbound_nodes'][0]] 

            layers_deep[layer['name']] = max([layers_deep[layer] for layer in list_of_preveous_layers]) + 1

        else:

            layers_deep[layer['name']] = layers_deep[layer['inbound_nodes'][0][0][0]] + 1


    return layers_deep



def get_layer_distribution_and_count (model: Model, layer_class_name: str, deep_function:function = max_neural_net_deep) -> tuple:
    """
        This function get the distribution and the count of layers in the model. 

        Receives: 
            model: Keras model

            layer_class_name: The layer class name in the model

            deep_function: The function that defines the deep of each layer and returns a dict containing {'{layer_name}' : deep, ...}

        Returns: 
            A pair containing: count, distribution
       
        Raises: 
            Nothing
    """

    layers_config: dict = json.loads(model.to_json())['config']['layers']

    nodes_depp: dict = deep_function(model)

    count: int = 0

    distribution: float = 0

    for layer in layers_config:

        if layer['class_name'] == layer_class_name:
            count += 1
            distribution += nodes_depp[layer['name']]

    if count:
        distribution = distribution/(count*max(nodes_depp.values()))

    return count, distribution



def get_number_of_layers_with_params (model) -> int:
    """
        Return the number of layers that have treinable parameters.

        Receives: 
            model: Keras model

        Returns: 
            A pair containing: count, distribution
        
        Raises: 
            Nothing
    """
    total_layers = 0

    for layer in model.layers:

        if layer.count_params() > 0:

            total_layers += 1
            
    return total_layers


def get_initializer_info(model) -> dict:
    """
        Get the intitializer type count for bias and kernel

        Receives: 
            model: Keras model

        Returns: 
            Dict containing the count of each initializer type for kernel and bias
        
        Raises:
            Nothing
    """

    count_of_initializers: dict = {
        "kernel" : {},
        "bias" : {},
        }

    kernel_initializer_type: dict = {
    }

    bias_initializer_type: dict = {
    }

    layers_config: dict = json.loads(model.to_json())['config']['layers']

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


def get_parameters_distribution(model: Model, deep_function:function = max_neural_net_deep) -> int:
    """
        Get the distribution of parameters along a keras model.

        Receives: 
            model: Keras model

            deep_function: The function that defines the deep of each layer, and returns a dict containing {'{layer_name}' : deep, ...}

        Returns: 
            float describing the distribution
        
        Raises:
            Nothing
    """
    nodes_depp = deep_function(model)

    distribution: float = 0.0
    
    for layer in model.layers:

        distribution += layer.count_params()*nodes_depp[layer.name] 
        
    distribution = distribution/(model.count_params()*model.layers.__len__()) # Normalization
        
    return distribution



def get_paralelism_info(model: Model):
    """
        Get the number of paralel sequences of neurons, with the number of inbound nodes of the concatenate layers.

        Obs: May not work if the model have a paralelism not maked by concatenate layers.

        Receives: 
            model: Keras model

        Returns: 
            The count of paralel segments.
        
        Raises:
            Nothing
    """

    info: dict = {
        'number' : 1
    }

    layers_config: dict = json.loads(model.to_json())['config']['layers']
    
    for layer in layers_config:

        if layer['class_name'] == "Concatenate":
            info['number'] += layer['inbound_nodes'][0].__len__() - 1

    return info['number']


def get_number_of_regularizers(model, regularizer_type: str = "") -> dict:
    """
        Get the count of regularizers for kernel, bias and activity in the model layers.

        Receives:
            model: Keras model

            regularizer_type: the regularizer class name, or "" for all types

        Returns: 
            the count of the regularizer
        
        Raises:
            Nothing
    """
        
    layers_config: dict = json.loads(model.to_json())['config']['layers']

    number_of_regularizers: dict = {
        "kernel" : 0,
        "bias" : 0,
        "activity" : 0
    }

    for layer in layers_config:
        
        if 'kernel_regularizer' in layer['config']: # Is possible that the layer haven't a kernel_regularizer option

            if (layer['config']['kernel_regularizer'] != None): # The layer have a regularizer  
                
                if regularizer_type == "":
                    number_of_regularizers["kernel"] += 1
                
                elif layer['config']['kernel_regularizer']['class_name'] == regularizer_type:
                    number_of_regularizers["kernel"] += 1
        
        if 'bias_regularizer' in layer['config']:

            if (layer['config']['bias_regularizer'] != None):
                
                if regularizer_type == "":
                    number_of_regularizers["bias"] += 1
                
                elif layer['config']['bias_regularizer']['class_name'] == regularizer_type:
                    number_of_regularizers["bias"] += 1
        
        
        if 'activity_regularizer' in layer['config']:

            if (layer['config']['activity_regularizer'] != None):
                
                if regularizer_type == "":
                    number_of_regularizers["activity"] += 1
        
                elif layer['config']['activity_regularizer']['class_name'] == regularizer_type:
                    number_of_regularizers["activity"] += 1

    return number_of_regularizers


def get_regularizer_distribution (model, regularizer_type: str = "", deep_function: function = max_neural_net_deep) -> dict:

    """
        Get the distribution of regularizers for kernel, bias and activity in the model layers.

        Receives: 
            model: Keras model

            regularizer_type: the regularizer class name, or "" for all types

            deep_function: The function that defines the deep of each layer and returns a dict containing {'{layer_name}' : deep, ...}

        Returns: 
            Dict with kernel, bias, activity, and the distribution of the regularizer type respectively {'kernel': 0.45, ...}
        
        Raises:
            Nothing
    """

    layers_config: dict = json.loads(model.to_json())['config']['layers']

    distribution_of_regularizers: dict = {
        "kernel" : 0.0,
        "bias" : 0.0,
        "activity" : 0.0
    }

    nodes_depp = deep_function(model)


    for layer in layers_config:

        if 'kernel_regularizer' in layer['config']: # Is possible that the layer haven't a kernel_regularizer option

            if (layer['config']['kernel_regularizer'] != None): # The layer have a regularizer  
                
                if regularizer_type == "":
                    distribution_of_regularizers["kernel"] += nodes_depp[layer['name']]
                
                elif layer['config']['kernel_regularizer']['class_name'] == regularizer_type:
                    distribution_of_regularizers["kernel"] += nodes_depp[layer['name']]
        
        if 'bias_regularizer' in layer['config']:

            if (layer['config']['bias_regularizer'] != None):
                
                if regularizer_type == "":
                    distribution_of_regularizers["bias"] += nodes_depp[layer['name']]
                
                elif layer['config']['bias_regularizer']['class_name'] == regularizer_type:
                    distribution_of_regularizers["bias"] += nodes_depp[layer['name']]
        
        
        if 'activity_regularizer' in layer['config']:

            if (layer['config']['activity_regularizer'] != None):
                
                if regularizer_type == "":
                    distribution_of_regularizers["activity"] += nodes_depp[layer['name']]
        
                elif layer['config']['activity_regularizer']['class_name'] == regularizer_type:
                    distribution_of_regularizers["activity"] += nodes_depp[layer['name']]
        
    max_depp = (max(nodes_depp.values()))

    total_regularizers = get_number_of_regularizers(model, regularizer_type)

    if total_regularizers['activity'] > 0:
        distribution_of_regularizers['activity'] = distribution_of_regularizers['activity']/( (max_depp)*(total_regularizers['activity']))

    if total_regularizers['bias'] > 0: 
        distribution_of_regularizers['bias'] = distribution_of_regularizers['bias']/( (max_depp)*(total_regularizers['bias']))

    if total_regularizers['kernel'] > 0:
        distribution_of_regularizers['kernel'] = distribution_of_regularizers['kernel']/( (max_depp)*(total_regularizers['kernel']) )

    return distribution_of_regularizers



def get_parameters_variation_info (model: Model, deep_function: function = max_neural_net_deep):
    """
        Get the variation information of the parameters of the model

        Receives: 
            model: Keras model

        Returns: 
            Dict with kernel, bias, activity, and the distribution of the regularizer type respectively {'kernel': 0.45, ...}
        
        Raises:
            Nothing
    """
    pass


    