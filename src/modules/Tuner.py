from sys import path
from os import getcwd, environ, walk

environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras_tuner import Objective, BayesianOptimization, Hyperband, RandomSearch, Tuner
from modules.DataMod import DataSet

class ModelTuner():

    def __init__(self,
                model_function,
                dataset : DataSet,
                objective : Objective,
                num_evaluate_epochs : int,
                evaluate_batch_size : int,
                tuner : Tuner = None,
                tuner_id : str = "Hyperband",
                tuner_params : dict = None,
                models_reclamation : float = 0.2,
                output_logs_path : str = None,
                callbacks : list = None):

        if tuner is None:
            self.__tuner = self.__set_tuner(tuner_id, model_function, objective, **tuner_params)            
            self.__trials = tuner_params["max_trials"]            
        else:
            self.__tuner = tuner

        self.__dataset = dataset
        self.__epochs = num_evaluate_epochs
        self.__batch_size = evaluate_batch_size

        self.__models_useful = int(self.__trials * models_reclamation) if tuner_params["max_trials"] else 1

        self.__logs_path = output_logs_path
        self.callbacks = callbacks
            
        return self

    def start_tunning(self, validation_perc : float = None, validation_data : DataSet = None ,callbacks = None):

        print(f"Tunning model {self.model_name}.")
        self.__tuner.search_space_summary()

        print("Starting tunning...")
        self.__tuner.search(self.__dataset.x_train, self.__dataset.y_train,\
                            validation_data = validation_data if validation_data else validation_perc,\
                            callbacks = callbacks)
        print("Tunning finished!")

        print("Printing results summary...")
        self.tuner.results_summary(num_trias=self.__trials)

    def __set_tuner(self, tuner_id, model_function, objective, **kwargs) -> Tuner:

        if tuner_id == "Hyperband":
            return Hyperband(model_function, objective= objective, **kwargs)
        elif tuner_id == "BayesianOptimization":
            return BayesianOptimization(model_function, objective= objective, **kwargs)
        elif tuner_id == "RandomSearch":
            return RandomSearch(model_function, objective= objective, **kwargs)
        else:
            raise ValueError("Tuner type not recognized.")