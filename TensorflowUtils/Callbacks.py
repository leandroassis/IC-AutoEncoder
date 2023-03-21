from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.python.lib.io.file_io import file_exists

from pandas import DataFrame, read_csv

from typing import Callable

import numpy as np


class MultipleTrainingLogger(Callback):
    """

    """
    def __init__(self, 
                 stop_function: Callable[[list, str, Callable], bool],
                 metric_name: str,
                 dir_name: str = 'Logs',
                 trainings_csv_name: str = 'Trainings_data',
                 models_csv_name:str = 'Models_data',
                 
                   ):
        super(MultipleTrainingLogger, self).__init__()
        
        self.per_epoch_batch_results: dict = {}
        self.epoch_mean_results: dict = {}

        self.stop_function: Callable[[list, str, Callable], bool] = stop_function
        self.metric_name: str = metric_name
        self.stoped_epoch: int = 0
        self.training_stoped: bool = False

        self.dir_name = dir_name

        self.logger_states = self.get_states()
    
    # Geters

    def get_optimizer_kwargs(self) -> dict: 
        return self.model.optimizer.get_config()

    def get_model_name(self) -> str:
        return self.model.name

    def get_loss_kwargs(self) -> dict:

        if isinstance(self.model.loss, Loss):
            return self.model.loss.name

    def get_states (self) -> dict:
        pass
    # Operations

    def append_results(self, results_dict: dict, logs: dict):
        
        for metric_name, metric_value in logs.items():
            
            if not metric_name in results_dict:
                results_dict[metric_name] = []
            
            results_dict[metric_name].append(metric_value)

    def append_epoch_mean (self):

        for metric_name, metric_array in self.per_epoch_batch_results.items():
            
            metric_epoch_mean =  np.mean(metric_array, axis = 0)
            
            if not metric_name in self.epoch_mean_results:
                self.epoch_mean_results[metric_name] = []
            
            self.epoch_mean_results[metric_name].append(metric_epoch_mean)

    
    def write_data_to_table (self, columns_and_values: dict, unique_identifier: str, table_path:str, write_over: bool = True) -> None:
        """
            Obs: All the dict data has to be str or number.

            columns_and_values:
                A dict with label and data

            unique_identifier:
                Column name that has unique values

            table_path:
                Path of csv file to save and load
        """
        new_data = DataFrame([columns_and_values])
        new_data.set_index(unique_identifier, inplace=True)

        if file_exists(table_path):
            dataframe = read_csv(table_path, index_col=0)
        else:
            dataframe = DataFrame(columns = columns_and_values.keys(), index=unique_identifier)

        if dataframe.empty:
            dataframe = dataframe.append(new_data)

        else: 
            if columns_and_values[unique_identifier] in dataframe.index and write_over: # checks if the table has data with the same unique identifier
                index = columns_and_values.pop(unique_identifier)
                dataframe.loc[dataframe.index == index, list(columns_and_values.keys())] = list(columns_and_values.values())
                # and subscribe in the line
            else:
                if not (columns_and_values[unique_identifier] in dataframe.index):
                    dataframe = dataframe.append(new_data)

        if dataframe.index.name != unique_identifier:
            dataframe.set_index(unique_identifier, inplace=True)
        dataframe.to_csv(table_path)
    

    def save_states(self):
        pass

    # Batch and epoch operations
    
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        
        # clean epoch data
        self.per_epoch_batch_results: dict = {}
    
    def on_batch_end(self, batch, logs=None):
        
        self.append_results(self.per_epoch_batch_results, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        
        self.append_epoch_mean()

        self.model.stop_training = self.stop_function(self.epoch_mean_results, self.metric_name)
        
        self.stoped_epoch = epoch
    
    def on_train_end(self, logs=None):
        return super().on_train_end(logs)
    

