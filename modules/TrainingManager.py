"""
    Training Manager module.
    ========================


    Explicação
"""
import sys, os

sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder'))
sys.path.insert(0, os.path.abspath('/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/modules'))

from TrainingData import KerasTrainingData
from CsvWriter import CsvWriter
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import Callback, CSVLogger
from tensorflow.keras.models import Model, load_model

from typing import Generator, List
from tensorflow.python.lib.io.file_io import file_exists

from pandas import read_csv

from NeuralNetData import KerasNeuralNetData
from DataMod import DataSet
from DirManager import KerasDirManager
from TensorBoardWriter import TensorBoardWriter


class TrainingManagerABC (ABC):
    """
        Training base class
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_csv_data():
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
                 new = True
                 ) -> None:
        """
        
        
        """
        self.training_idx = self._get_training_idx(new)
        
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
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        
        self.callbacks = callbacks
        
        if isinstance(dataset, str):
            self.dataset = DataSet.load_by_name(dataset)
        else:
            self.dataset = dataset

        KerasDirManager.__init__(self, model_name = self.neural_net_data.model_name, 
                                dataset_name = self.dataset.name,
                                training_idx = self.training_idx,
                                loss_name = loss.__name__)

        CsvWriter.__init__(self, file_name = "AllTrainingData", training_idx = self.training_idx)

        TensorBoardWriter.__init__(self, file_path= self.logs_dir)

        KerasTrainingData.__init__(self)

        if new:
            self.remove_last_save()

        self.generators: dict = None

        self._get_generator_atributes()

        self.make_all_dirs()


    def _get_model (self):
        """
        
        """
        if file_exists(self.model_save_pathname):
            custom_objs: dict = {}

            for metric in self.metrics:
                custom_objs[metric.__name__] = metric

            for callback in self.callbacks: 
                custom_objs[callback.__class__.__name__] = callback

            custom_objs[self.loss.__name__] = self.loss(**self.loss_kwargs)

            custom_objs[self.optimizer.__name__] = self.optimizer(**self.optimizer_kwargs)

            model = load_model(self.model_save_pathname, custom_objects = custom_objs, compile = False)

            model.compile(optimizer = self.optimizer(**self.optimizer_kwargs),
                           loss = self.loss(**self.loss_kwargs),
                           metrics = self.metrics,
                           **self.compile_kwargs)

            print("---> Model loaded")

            return model

        else:
            return self.neural_net_data.model
        

    def _get_generator_atributes(self) -> None:
        """
            Get all atributes if they are a generator or a list

            Parameters:
                None

            Returns:
                None

            Raise:
                Nothing
        """

        generators: dict = {}

        for key, value in self.__dict__.items():
            
            if value.__class__.__name__ == "generator":
                
                generators[key] = list(value)

            elif value.__class__.__name__ == "list" and value:
                if value[0] == "generator":
                    value.pop(0)
                    generators[key] = value

        self.generators = generators


    def _make_training_steps(self, mode:str = "all_combinations") -> Generator:
        """
            Generator for training sequences
        """

        if mode == "all_combinations":
            keys, values = self.generators.items()

            return self._recursive_for_iterator(keys, values)

        if mode == "sequence":
            pass


    def _sequence_iterator(self) -> Generator:

        for key, value in self.generators.items():

            self.__dict__[key] = value[0]
            
        
    
    def _recursive_for_iterator (self, keys, values, idx = 0) -> Generator:
        """

        """
        for item in values[idx]:

            self.__dict__[keys[idx]] = item
            self._recursive_updater()


            if idx < len(keys):
                self.recursive_for(self, idx + 1)
            else:
                yield

    """
    def recursive_updater (dictionary:dict, Kw):
        for key, value in Kw.items():
            # all this is to check if the atribute of type 'dict' will be replaced, or have some key values changed.
            if type(value) == dict:
                if key[0] == '*': # replace the dict
                    dictionary[key[1:]] = value
                else:
                    recursive_updater(dictionary[key], value) # change key values.
            else:
                dictionary[key] = value
    """

    def start_training(self) -> None:
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
        
        x_train = self.dataset.x_train
        x_test = self.dataset.x_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test
        

        neural_net: Model = self._get_model()    
            
            

        neural_net.compile(optimizer = self.optimizer(**self.optimizer_kwargs),
                           loss = self.loss(**self.loss_kwargs),
                           metrics = self.metrics,
                           **self.compile_kwargs)

        self.make_all_dirs()

        csv_logger = CSVLogger(filename = self.csv_pathname, separator = ';', append= True)

        self.callbacks.append(csv_logger)
    
        last_epoch = self._get_last_epoch_()

        self.fit_kwargs['epochs'] += 1 + last_epoch

        neural_net.fit(x = x_train, y = y_train,
                       validation_data = (x_test, y_test),
                       initial_epoch = last_epoch + 1,
                       callbacks = self.callbacks,
                       **self.fit_kwargs)


        neural_net.save(filepath = self.model_save_pathname)

        self.save_actual_model_pathname_as_last()

        self.write_data_to_table(self.get_csv_data())

        image_cluster = self.get_example_imgs(neural_net)

        descrptions = ["Inputs", "Outputs", "Expected", "Gaussian_filter"]

        for idx in range (4):
            self.write_images(image_cluster[idx], descrptions[idx])


    def _get_training_idx(self, new):

        add = 0

        if new:
            add = 1
        
        if file_exists("logs/AllTrainingData.csv"):
            dataframe = read_csv("logs/AllTrainingData.csv")
        else:
            return 0

        return len(dataframe.index) - 1 + add



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


    def get_csv_data (self):
        """
        
        """
        data = self.neural_net_data.get_csv_data()

        training_params: dict = {
            "training_idx" : self.training_idx,
            "optimizer" : self.optimizer.__name__,
            "optimizer_args" : self.optimizer_kwargs,
            "loss" : self.loss.__name__,
            "loss_args" : self.loss_kwargs,
            "compile_kwargs" : self.compile_kwargs,
            "fit_kwargs" : self.fit_kwargs,
            "dataset" : self.dataset.name,
            "dataset_description" : self.dataset.description,
            "dataset_params" : self.dataset.parametros
        }

        training_results = self.get_best_results()

        data.append(training_params)
        data.append(training_results)

        return data

