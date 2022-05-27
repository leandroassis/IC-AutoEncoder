"""
    Training Manager module.
    ========================


    Explicação
"""
import sys, os
import pickle

sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder'))
sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/modules'))

from TrainingData import KerasTrainingData
from CsvWriter import CsvWriter
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.keras.models import Model, load_model
from copy import deepcopy

from typing import Generator, List
from tensorflow.python.lib.io.file_io import file_exists

from pandas import read_csv

from NeuralNetData import KerasNeuralNetData
from DataMod import DataSet
from DirManager import KerasDirManager
from TensorBoardWriter import TensorBoardWriter

from glob import glob
from misc import get_loss_name, get_model, get_current_time_and_data

class function:
    pass

class TrainingManagerABC (ABC):
    """
        Training base class
    """

    @abstractmethod
    def __init__(self) -> None:
        pass


    @abstractmethod
    def start_training() -> None:
        pass
    


class KerasTrainingManager (TrainingManagerABC, 
                            KerasDirManager,
                            CsvWriter,
                            KerasTrainingData,
                            TensorBoardWriter):
    """
        Keras training manager
    """
    
    def __init__(self,
                 neural_net_data: KerasNeuralNetData = None,
                 optimizer: Optimizer = None,
                 optimizer_kwargs: dict = None,
                 loss: Loss = None,
                 loss_kwargs: dict = None,
                 metrics: List[str] = None,      
                 compile_kwargs: dict = None,
                 fit_kwargs: dict = None,
                 callbacks: List[Callback] = [],
                 dataset: DataSet = None,
                 training_function: function = None,
                 training_idx: int = None,
                 best_selector_metrics: int = None
                 ) -> None:
        """
        
        
        """

        if training_idx == None:

            if not training_function:
                raise Exception("Training function not set")

            self.training_idx = self._get_training_idx()
            
            if isinstance(neural_net_data, str):
                self.neural_net_data = KerasNeuralNetData(model_name = neural_net_data, load_model= True)
            elif isinstance(neural_net_data, Model):
                self.neural_net_data = KerasNeuralNetData(model = neural_net_data)
            else:
                self.neural_net_data = neural_net_data
            
            
            self.optimizer = optimizer
            self.optimizer_kwargs = optimizer_kwargs
            self.loss = loss
            self.loss_kwargs = loss_kwargs
            self.metrics = metrics
            self.best_selector_metrics = best_selector_metrics
            self.compile_kwargs = compile_kwargs
            self.fit_kwargs = fit_kwargs
            
            self.callbacks = callbacks

            self.training_function = training_function
            
            if isinstance(dataset, str):
                self.dataset = DataSet.load_by_name(dataset)
            else:
                self.dataset = dataset

        else:

            self.optimizer = None
            self.optimizer_kwargs = None
            self.loss = None
            self.loss_kwargs = None
            self.compile_kwargs = None
            self.fit_kwargs = None
            self.training_function = None
            self.metrics = None
            self.dataset = None
            self.best_selector_metrics = None
            self.callbacks = None
            self.training_idx = training_idx

            self.load_state()

        self.loaded = False

        KerasDirManager.__init__(self, model_name = self.neural_net_data.model_name, 
                                dataset_name = self.dataset.name,
                                training_idx = self.training_idx,
                                loss_name = get_loss_name(self.loss))

        CsvWriter.__init__(self, file_name = "AllTrainingData", training_idx = self.training_idx)

        TensorBoardWriter.__init__(self, file_path= self.logs_dir)

        KerasTrainingData.__init__(self)

        self.make_all_dirs()

          

    def _get_model (self):
        """
        
        """
        if file_exists(self.model_save_pathname):
            custom_objs: dict = {}

            if self.metrics:
                for metric in self.metrics:
                    if not isinstance(metric, str):
                        custom_objs[metric.__name__] = metric

            if self.callbacks:
                for callback in self.callbacks: 
                    custom_objs[callback.__class__.__name__] = callback

            custom_objs[self.optimizer.__name__] = self.optimizer(**self.optimizer_kwargs) if self.optimizer_kwargs else self.optimizer()

            model = load_model(self.model_save_pathname, compile = False)

            print("---> Model loaded")

            self.neural_net_data.model = model

            self.loaded = True
        

   

    def start_training(self, epochs = None) -> None:
        """
            Initializes the training

            Parameters
            ----------

            None

            Returns
            -------

            None

            Raise
            -----

            Nothing
        """
        print(f"---> Initiating training number: {self.training_idx}")

        self.training_function(self, epochs = epochs)

        self.write_data_to_table(self.get_training_csv_data(), "training_idx", self.trainings_data_path)

        self.write_data_to_table(self.neural_net_data.get_model_csv_data() , "model_name", self.models_data_path)

        """
        image_cluster = self.get_example_imgs(self._get_model())

        descrptions = ["Inputs", "Outputs", "Expected", "Gaussian_filter"]

        for idx in range (4):
            self.write_images(image_cluster[idx], descrptions[idx])
        """
        self.save_state()


    def _get_training_idx(self):
        
        if file_exists("logs/AllTrainingData.csv"):
            dataframe = read_csv("logs/AllTrainingData.csv")
        else:
            return 0

        return len(dataframe.index)



    def _get_last_epoch_(self) -> int:
        """
            ## Função:

            Retorna a ultima época treinada de um checkpoint. \n
            obs: retorna -1 quando nenhum treino foi realizado para o treino inciar na época 0.
        """
        last_epoch:int

        if file_exists(self.csv_pathname) and os.path.getsize(self.csv_pathname) > 0:
            dataframe = self.get_csv_training_history()
            
            if dataframe.empty:
                last_epoch = -1
            else:
                last_epoch = dataframe["epoch"].tolist()[-1]
        else:
            last_epoch = -1

        return last_epoch


    def get_training_csv_data (self):
        """
        
        """
        date, time = get_current_time_and_data()

        training_results = self.get_best_results()

        training_params_and_data: dict = {
            "training_idx" : int(self.training_idx),
            "model_name" : self.neural_net_data.model_name,
            "optimizer" : self.optimizer.__name__,
            "optimizer_kwargs" : self.optimizer_kwargs,
            "loss" : get_loss_name(self.loss),
            "loss_kwargs" : self.loss_kwargs,
            "compile_kwargs" : self.compile_kwargs,
            "fit_kwargs" : self.fit_kwargs,
            "dataset" : self.dataset.name,
            "dataset_description" : self.dataset.description,
            "dataset_params" : self.dataset.parameters,
            "results": training_results,
            "date": date,
            "time": time,
        }

        return training_params_and_data
    
    def save_state(self) -> None:
        '''
           Saves the state of the attributes in the self.attributes_save_pathname
        '''
        print("--> Saving the actual state.")

        attributes_to_save = [self.optimizer,
                            self.optimizer_kwargs,
                            self.loss,
                            self.loss_kwargs,
                            self.compile_kwargs,
                            self.fit_kwargs,
                            self.training_function,
                            self.metrics,
                            self.dataset.name,
                            self.dataset.parameters,
                            self.best_selector_metrics,
                            self.callbacks]

        attributes_names = ["optimizer",
                            "optimizer_kwargs",
                            "loss",
                            "loss_kwargs",
                            "compile_kwargs",
                            "fit_kwargs",
                            "training_function",
                            "metrics",
                            "dataset_name",
                            "dataset_parameters",
                            "best_selector_metrics",
                            "callbacks"]

        variables_to_save:dict = {}        

        for att, att_name in zip(attributes_to_save, attributes_names):
            if att:
                variables_to_save.update( {att_name: deepcopy(att)})

        with open(self.attributes_save_pathname, 'wb') as file:
            pickle.dump(variables_to_save, file, pickle.HIGHEST_PROTOCOL)
            file.close()
        
        print("--> Actual state saved.")


    def load_state(self) -> None:
        '''
            Loads the training_idx attributes from the file
        '''
        print("--> Loading the actual state.")

        path = glob(f"logs/**/{self.training_idx}/KTM_attributes.pkl", recursive = True)

        if path.__len__() != 1:
            Exception(f"Fail to load previous attributes. {path.__len__()} paths found")

        self.neural_net_data = KerasNeuralNetData(model = get_model(training_idx = self.training_idx))

        attributes_save_pathname = path[0]

        attributes_names = ["optimizer",
                            "optimizer_kwargs",
                            "loss",
                            "loss_kwargs",
                            "compile_kwargs",
                            "fit_kwargs",
                            "training_function",
                            "metrics",
                            "best_selector_metrics",
                            "callbacks"]
        
        with open(attributes_save_pathname, 'rb') as file:
            
            attributes:dict = pickle.load(file)
            
            for att_name in attributes_names:
                
                if att_name in attributes.keys():
                    self.__dict__[att_name] = attributes[att_name]

            if "dataset_parameters" in attributes.keys():
                self.dataset = DataSet().load_by_name(attributes["dataset_name"], attributes["dataset_parameters"])
            else:
                self.dataset = DataSet().load_by_name(attributes["dataset_name"])

            file.close()

        
    def change_parameters ( self,
                            neural_net_data: KerasNeuralNetData = None,
                            optimizer: Optimizer = None,
                            optimizer_kwargs: dict = None,
                            loss: Loss = None,
                            loss_kwargs: dict = None,
                            metrics: List[str] = None,      
                            compile_kwargs: dict = None,
                            fit_kwargs: dict = None,
                            callbacks: List[Callback] = [],
                            dataset: DataSet = None,
                            training_function: function = None,
                            training_idx: int = None,
                            best_selector_metrics = None):

        if neural_net_data:
            self.neural_net_data = neural_net_data

        if optimizer:
            self.optimizer = optimizer

        if optimizer_kwargs:
            self.optimizer_kwargs = optimizer_kwargs

        if loss:
            self.loss = loss

        if loss_kwargs:
            self.loss_kwargs = loss_kwargs

        if metrics:
            self.metrics = metrics

        if compile_kwargs:
            self.compile_kwargs = compile_kwargs

        if callbacks:
            self.callbacks = callbacks

        if dataset:
            self.dataset = dataset
        
        if training_function:
            self.training_function = training_function
        
        if training_idx != None:
            self.training_idx = training_idx

        if best_selector_metrics:
            self.best_selector_metrics = best_selector_metrics

        if fit_kwargs:
            self.fit_kwargs = fit_kwargs