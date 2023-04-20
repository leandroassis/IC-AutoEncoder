from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.python.lib.io.file_io import file_exists
from TensorflowUtils.NeuralNetData import KerasNeuralNetData
from TensorflowUtils.DirManager import KerasDirManager
from TensorflowUtils.DataSet import DataSetABC


from pandas import DataFrame, read_csv

from typing import Callable

import numpy as np
import json

class MultipleTrainingLogger(Callback):
    """
    A Callback to collect and save data of multiple trainings. It also have a 
    early stopping feature to ensure that all models are trained until satisfy a 
    condition.

    ## Colected Data

    ### To trainings data table

    - Model name
    - Optimizer config
    - Loss name/config
    - Best metrics results and respective epoch
    - Data set name
    - stop training method

    ### To models data table

    - Model name
    - Total number of parameters
    - Model number of layers

    ### To the training folder

    - Metrics and loss mean for all epochs.
    - Output samples

    ## Parâmeters

    stop_function
    metric_name
    dir_name
    trainings_csv_name
    models_csv_name
    data_set
    """

    def __init__(self, 
                 stop_function: Callable[[list, str, Callable], bool],
                 monitor_metric_name: str,
                                  
                 dir_name: str = 'Logs',
                 trainings_csv_name: str = 'Trainings_data',
                 models_csv_name:str = 'Models_data',
                 
                 data_set:object = None,

                ):
        super(MultipleTrainingLogger, self).__init__()

        self.data_set = data_set

        # Metrics evaluation attributes
        
        self.per_epoch_batch_results: dict = {}
        self.epoch_mean_results: dict = {}

        # Early Stoping attributes

        self.stop_function: Callable[[list, str, Callable], bool] = stop_function
        self.monitor_metric_name: str = monitor_metric_name
        self.stoped_epoch: int = 0

    
        # Directory attributes

        self.dir_manager = None
        self.models_csv_name = models_csv_name
        self.dir_name = dir_name
        self.trainings_csv_name = trainings_csv_name

        # Data collectors

        self.model_data_collector = None

        # states

        self.states_path = f"{dir_name}/states.json"
        self.states = self.get_states()

        
    
    # Geter methods

    def get_optimizer_kwargs(self) -> dict: 
        return self.model.optimizer.get_config()

    def get_model_name(self) -> str:
        return self.model.name

    def get_loss_kwargs(self) -> dict:

        if isinstance(self.model.loss, Loss):
            return self.model.loss.__dict__
        
        if issubclass(self.model.loss, Loss):
            return self.model.loss().__dict__
        
        return {}

    def get_states (self) -> dict:

        if file_exists(self.states_path):
            
            with open(self.states_path, "r") as json_file:
                json_str = json_file.read()
                json_file.close()
                
            return json.load(json_str) 
        
        return {"training_idx": 0}

    def get_best_results (self):

        best_results = {}

        for metric in self.epoch_mean_results.keys():

            if metric == "loss" or metric == "val_loss":

                best_results[metric] = min(self.epoch_mean_results[metric])
            
            else:
                best_results[metric] = max(self.epoch_mean_results[metric])

        return best_results

    def get_data_set_kwargs (self):

        if self.data_set == None:
            return {"No data"}
        
        if isinstance(self.data_set, DataSetABC):
            return self.data_set.get_metadata()
        
        return  {"No data"}
    
    # Operation methods

    def _append_results(self, results_dict: dict, logs: dict):
        
        for metric_name, metric_value in logs.items():
            
            if not metric_name in results_dict:
                results_dict[metric_name] = []
            
            results_dict[metric_name].append(metric_value)

    def _append_epoch_mean (self):

        for metric_name, metric_array in self.per_epoch_batch_results.items():
            
            metric_epoch_mean =  np.mean(metric_array, axis = 0)
            
            if not metric_name in self.epoch_mean_results:
                self.epoch_mean_results[metric_name] = []
            
            self.epoch_mean_results[metric_name].append(metric_epoch_mean)

    
    def _write_data_to_table (self, columns_and_values: dict, unique_identifier: str, table_path:str, write_over: bool = True) -> None:
        """
            Obs: All the dict data has to be str or number.

            columns_and_values:
                A dict with label and data

            unique_identifier:
                Column name that has unique values

            table_path:
                Path of the csv file to save and load
        """
        new_data = DataFrame([columns_and_values])
        
        new_data.set_index(unique_identifier, inplace=True)

        if file_exists(table_path):
            dataframe = read_csv(table_path, index_col=0)
        else:
            dataframe = DataFrame(columns = columns_and_values.keys())
            dataframe.set_index(unique_identifier, inplace=True)

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

    def _write_data_to_file (self, columns_and_values: dict, unique_identifier:str, table_path) -> None:
        """
        Write the dict to csv where the keys are the csv columns and every 
        line contais elements off the dict value (probably a list)
        """

        dataframe = DataFrame(columns_and_values)
        dataframe.set_index(unique_identifier, inplace=True)
        dataframe.to_csv(table_path)

    
    def progress_in_metric (self, metric_name):

        if self.epoch_mean_results[metric_name].__len__() <= 1:
            return True

        last_result = self.epoch_mean_results[metric_name][-1]
        penultimate_result = self.epoch_mean_results[metric_name][-2]

        if last_result > penultimate_result:
            return True
        
        return False

    def _save_as_json(self, data, file_path):
        """
        Saves a dict in a .json file
        """
        json_str = json.dumps(data, indent=4)
        
        with open(file_path, "w+") as json_file:
            json_file.write(json_str)
            json_file.close()
        
    def _save_states(self):
        """
        Save the states of the trainings logger
        """
        self.states["training_idx"] += 1
        self._save_as_json(self.states, self.states_path)


    def generate_training_data (self) -> dict:
        """
        returns the data that'll be save on csv file
        """
        training_data: dict = {
            "training_idx": self.states["training_idx"],
            "model_name": self.get_model_name(),
            "optimizer_args": self.get_optimizer_kwargs(),
            "loss_args": self.get_loss_kwargs(),
            "data_set_info": self.get_data_set_kwargs(),
            "best_results": self.get_best_results(),
        }

        return training_data 

    # Batch and epoch methods
    
    def on_train_begin(self, logs=None):

        self.dir_manager = KerasDirManager(logs_dir_name= self.dir_name,
                                           model_name=self.get_model_name(),
                                           trainings_csv_name= self.trainings_csv_name,
                                           models_csv_name = self.models_csv_name,
                                           dataset_name = self.data_set.get_metadata()["name"],
                                           training_idx = self.states["training_idx"]
                                           )

        self.model_data_collector = KerasNeuralNetData(self.model)


    def on_epoch_begin(self, epoch, logs=None):
        # clean epoch data
        self.per_epoch_batch_results: dict = {}
    

    def on_train_batch_end(self, batch, logs=None):
        
        self._append_results(self.per_epoch_batch_results, logs)

    def on_test_batch_end (self, batch, logs):

        val_logs = {}

        for metric in logs.keys():

            val_logs["val_" + metric] = logs[metric]

        self._append_results(self.per_epoch_batch_results, val_logs)
    
    def on_epoch_end(self, epoch, logs=None):
        
        self._append_epoch_mean()

        self.model.stop_training = self.stop_function(self.epoch_mean_results, self.monitor_metric_name)
        
        self.stoped_epoch = epoch

        if self.progress_in_metric(self.monitor_metric_name):
            self.model.save(self.dir_manager.model_save_best_training_pathname)

        if self.progress_in_metric(f"val_{self.monitor_metric_name}"):
            self.model.save(self.dir_manager.model_save_best_validation_pathname)

    
    def on_train_end(self, logs=None):
        
        self._write_data_to_table(self.generate_training_data(), 
                                  unique_identifier = "training_idx", 
                                  table_path = self.dir_manager.trainings_table_pathname)
        

        self.epoch_mean_results["epoch"] = list(range(1,self.stoped_epoch+2, 1))
        self._write_data_to_file(self.epoch_mean_results,
                                  unique_identifier="epoch",
                                  table_path = self.dir_manager.metric_means_pathname)
        
        self._write_data_to_table(self.model_data_collector.get_model_csv_data(),
                                  unique_identifier="model_name",
                                  table_path=self.dir_manager.models_table_pathname)

        self._save_states()