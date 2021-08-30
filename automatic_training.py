import pickle

from os.path import isdir
from os import makedirs
from pandas.io.parsers import read_csv
from tensorflow.keras.models import model_from_json, Model
from pandas import DataFrame
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.lib.io.file_io import file_exists

from misc import get_current_time_and_data, get_last_epoch

from DataMod import DataSet

class Training_State ():
    '''
        ## Function:

        This class saves the state of Auto_training objects.
    '''
    def set_state(self, model_name:str, training_idx:int, fit_Kwargs:dict, dataset_name:str, 
    compile_kwargs:dict, number_of_ephocs:int,  loss_class:Loss, loss_kwargs:dict, 
    optimizer_class:Optimizer, optimizer_kwargs:dict) -> None:
        """
            ## Function:

            Set a new state.

            ## Recieves:

            All states parameters.

            ## Returns:

            None
        """

        # Training params
        self.training_idx:int = training_idx
        self.model_name:str = model_name
        self.number_of_ephocs:int = number_of_ephocs
        self.fit_Kwargs:dict = fit_Kwargs

        # Dataset
        self.dataset_name:str = dataset_name

        # Time and date
        (self.date, self.time) = get_current_time_and_data()

        # Paths

        # Base dir
        self.sub_dir_0 = "Relatorios-Dados-etc/Resultados/"
        self.sub_dir_1 = self.dataset_name + '/'
        self.sub_dir_2 = self.model_name.replace('.json', '') + '/'
        self.sub_dir_3 = str(training_idx) + '/'
        self.data_path = self.sub_dir_0 + self.sub_dir_1 + self.sub_dir_2 + self.sub_dir_3

        # CSV_Logger
        csv_name:str = 'csv-#' + str(self.training_idx) + ".log"
        self.csv_pathname:str = self.data_path + csv_name

        # Model training name
        self.model_save_pathname:str = self.data_path + '#' + str(self.training_idx) + '-checkp'

        # Models directory
        self.models_dir:str = "nNet_models/"

        # Dataframe directory
        dataframe_name:str = "Results_DataFrame.df"
        self.dataframe_pathname:str = self.sub_dir_0 + dataframe_name

        # Dataframe params
        self.dataframe_columns:list = [ "training idx",
                                        "model name",
                                        "model total params",
                                        "model total layers",
                                        "optimizer",
                                        "optimizer args",
                                        "loss", 
                                        "loss args",
                                        "compile args",
                                        "best training loss",
                                        "best training epoch",
                                        "best validation loss",
                                        "best validation epoch",
                                        "last epoch training loss",
                                        "last epoch validation",
                                        "last epoch" ]
        
        # Training parameters
        self.last_epoch:int = get_last_epoch(self.csv_pathname)
        self.actual_epoch:int = self.last_epoch + self.fit_Kwargs['ephocs']
        self.loss:Loss = loss_class
        self.loss_kwargs:dict = loss_kwargs
        
        # Compiler parameters
        self.optimizer:Optimizer = optimizer_class
        self.optimizer_kwargs:dict = optimizer_kwargs
        self.compile_kwargs:dict = compile_kwargs


    def _update_dependent_atributes_ (self) -> None:
        '''
            Atualiza atributos que tem dependências de tempo ou de outros atibutos que podem mudar
        '''
        self.date, self.time = get_current_time_and_data()
        self.image_name = '#' + str(self.training_idx) + '|' + self.date + "|" + self.time + "|" + "|epoch=" + str(self.actual_epoch) + ".png"
        
        # diretorios base
        self.sub_dir_0 = "Relatorios-Dados-etc/Resultados/"
        self.sub_dir_1 = self.dataset.name + '/'
        self.sub_dir_2 = self.model_name.replace('.json', '') + '/'
        self.sub_dir_3 = self.date + '|' +self.time + '|' #' + str(training_idx) + '/'
        self.data_path = self.sub_dir_0 + self.sub_dir_1 + self.sub_dir_2 + self.sub_dir_3

        # CSV_Logger
        csv_name:str = 'csv-#' + str(self.training_idx) + ".log"
        self.csv_pathname:str = self.data_path + csv_name

        #checkpoints
        checkpoints_dir_path:str = "checkpoints/"
        checkpoint_dir_name:str = self.model_name.replace('.json', '-#' + str(self.training_idx)) + '-' + self.dataset.name + '/'
        checkpoint_name:str = '#' + str(self.training_idx) + '-checkp'
        self.checkpoint_pathname:str = checkpoints_dir_path + checkpoint_dir_name + checkpoint_name

    
    def change_atributes (self, Kw_att_and_val:dict) -> None:
        """
            ## Função:

            Muda determinados atributos especificadoes na forma de keyword e valor.

            ## Retorna:

            Sem retorno

            ## Exemplo:
            
            Caso queira mudar o nome do modelo a ser usado e do otimizador escreva:

            >>> Kw_att_and_val = {'model_name': 'nome', 'optimizer': optimizer_class}
            # note que 'model_name' e 'optimizer' são atributos de Auto_training
        """
        for key, value in Kw_att_and_val.items():
            self.__dict__[key] = value

        self._update_dependent_atributes_()


    def change_training_idx (self) -> None:
        self.training_idx += 1

        self._update_dependent_atributes_()


# ======================================================================================================================

# ======================================================================================================================

class Auto_Training ():
    """
        ## Função:

        A classe implementa o treino e a geração automatica de logs e resultados usando o keras e outras bibliotecas.
    """

    def __init__(self) -> None:
        self.version:str = '0.0.0'
        self.state_pathname:str = "Auto_Training_state/" + "state-v:" + self.version
        self.state:Training_State
        self.state.load_state(self.state_pathname)

    def save_state(self) -> None:
        '''
            ## Função :

            Salva o estado atual do objeto.
        '''
        with open(self.state_pathname, 'wb') as file:
            pickle.dump(self.state, file, pickle.HIGHEST_PROTOCOL)
            file.close()
        

    def load_state(self) -> None:
        '''
            ## Função :

            Carrega o estado armazenado no arquivo
        '''
        with open(self.state_pathname, 'rb') as file:
            preveous_obj:Training_State = pickle.load(file, pickle.HIGHEST_PROTOCOL)
            self.state = preveous_obj
            file.close()
        

   
    def _check_if_dirs_exists_ (self) -> None:
        """
            ## Função:

            Verifica se todos os diretórios que compoem os diretorios existem.
            E no caso de não existirem, o método cria esses diretórios.

            ## Exemplo: 
                * `Relatorios-Dados-etc/Imagens de resultados`

            São dois diretórios, ambos serão criados caso ja não existam na pasta
            onde o programa é executado.
        """

        # Checkpoints

        if not isdir(self.state.checkpoint_pathname):
            makedirs(self.state.checkpoint_pathname)

        if not isdir(self.state.data_path):
            makedirs(self.state.data_path)

        
        
    def _load_model_ (self) -> Model:
        '''
            ## Função:

            Carrega o modelo de rede neural definido no atributo `model_name`, 
            junto dos pesos armazenados no checkpoint definido no atibuto `checkpoint_name`
        '''

        # Tentando carregar o modelo

        if file_exists(self.state.model_save_pathname):
            nNet:Model = load_model(self.state.model_save_pathname)
            return nNet

        try:
            json_file = open(self.state.models_dir + self.model_name, "r")
        except FileNotFoundError:
            raise Exception("Fail atempt to load the model " + self.model_name + " at " + self.models_dir)

        json_readed = json_file.read()
        nNet:Model = model_from_json(json_readed)
        json_file.close()

        return nNet
    


    def get_csv_training_history (self) -> DataFrame:
        """

        """
        dataframe = read_csv(self.state.csv_pathname)
        return dataframe

    def get_best_results(self) -> dict:
        """
        
        """
        dataframe:DataFrame = self.get_csv_training_history ()

        best_training_loss = dataframe["loss"].min()
        best_training_epoch = (dataframe["loss"] == best_training_loss).index[0]

        best_val_loss = dataframe["val_loss"].min()
        best_val_epoch = (dataframe["val_loss"] == best_training_loss).index[0]

        last_training_loss = dataframe["loss"].tolist()[-1]
        last_validation_loss = dataframe["val_loss"].tolist()[-1]
        last_epoch = dataframe["epoch"].tolist()[-1]
        
        return {"best_training_loss": best_training_loss,
                "best_training_epoch" : best_training_epoch,
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
                "last_training_loss" : last_training_loss,
                "last_validation_loss": last_validation_loss,
                "last_epoch" : last_epoch}


    def _get_last_epoch_(self) -> int:
        """
            ## Função:

            Retorna a ultima época treinada de um checkpoint. \n
            obs: retorna -1 quando nenhum treino foi realizado para o treino inciar na época 0.
        """
        last_epoch:int

        if file_exists(self.state.csv_pathname):
            dataframe = self.get_csv_training_history()
            last_epoch = dataframe["epoch"].tolist()[-1]
        else:
            last_epoch = -1

        return last_epoch


    def save_data_to_dataframe(self) -> None:
        """
        
        """

        if not file_exists(self.state.dataframe_pathname):
            dataframe:DataFrame = DataFrame(columns=self.state.dataframe_columns)
            dataframe.to_csv(self.state.dataframe_pathname)

        
        # model parameters

        dataframe = read_csv(self.state.dataframe_pathname)

        model:Model = self._load_model_()
        number_of_model_parameters:int = model.count_params()
        number_of_model_layers:int = model.layers.__len__()
        model_name = self.state.model_name
        
        loss_data:dict = self.get_best_results()

        new_line = DataFrame(
            [{  "training idx": self.state.training_idx,
                "model name": self.state.model_name,
                "model total params": number_of_model_parameters,
                "model total layers": number_of_model_layers,
                "optimizer": self.state.optimizer._name,
                "optimizer args": self.state.optimizer_kwargs,
                "loss": self.state.loss.name, 
                "loss args": self.state.loss_kwargs,
                "compile args": self.state.compile_kwargs,
                "best training loss": loss_data["best_training_loss"],
                "best training epoch": loss_data["best_training_epoch"],
                "best validation loss": loss_data["best_val_loss"],
                "best validation epoch": loss_data["best_val_epoch"],
                "last epoch training loss": loss_data["last_training_loss"],
                "last epoch validation": loss_data["last_validation_loss"],
                "last epoch": loss_data["last_epoch"]
            }], 
        )

        new_dataframe_line = DataFrame(new_line)
        dataframe.append(new_dataframe_line)
        dataframe.to_csv(self.state.dataframe_pathname)



    def start_training(self) -> None:
        """
            ## Function:

                Starts the training

            ## Receives:

                Nothing

            ## Returns:

                None

            ## Examples:

                ...
               
            ## Raises:

                Nothing.
        """
        with self.state as stt:
            x_train = stt.dataset.x_train
            x_test = stt.dataset.x_test
            y_train = stt.dataset.y_train
            y_test = stt.dataset.y_test
        

        neural_net:Model = self._load_model_()

        neural_net.compile(optimizers = self.state.optimizer(**self.state.optimizer_kwargs),
                           loss=self.state.loss,
                           **self.state.compile_kwargs)

        self._check_if_dirs_exists_()

        csv_logger = CSVLogger(filename = self.state.csv_pathname, separator = ';')
        standard_callbacks:list = [csv_logger]
        
        neural_net.fit(x = x_train, y = y_train,
                       validation_data = (x_test, y_test),
                       initial_epoch = self._get_last_epoch_() + 1,
                       callbacks = standard_callbacks,
                       epochs = self.state.number_of_ephocs,
                       **self.state.fit_Kwargs)


        neural_net.save(filepath = self.state.model_save_pathname)


        self.save_data_to_dataframe()



    def set_a_new_training(self, new_parameters:dict) -> None:
        '''
            ## Function:

                Executes a new training after changes in parameters.

            ## Receives:

                A `dict` where the keys are the parameters to be changed, and the values are new parameters.

            ## Returns:

                None

            ## Examples:

                >>> self.set_a_new_training ( {"model_name": "model2",
                                           "dataset": new_dataset} )

            ## Raises:

                Nothing.
        '''
        self.state.change_training_idx()
        self.state.change_atributes(new_parameters)
        self.start_training()

    def set_a_sequence_of_trainings(self, list_of_changes:list) -> None:
        '''
            ## Function:

                Executes pieces of training after changes in parameters.

            ## Receives:

                A `list` where the elements are dicts containing the training changes.

            ## Returns:

                None

            ## Examples:

                >>> change1 = {"model_name": "model2"}
                >>> change2 = {"dataset": new_dataset}
                >>> list_of_changes = [change1, change2]
                >>> self.set_a_new_training ( list_of_changes )

                In this example, the first training begins after change1, and the second training is started afterward of change2.

            ## Raises:

                Nothing.
        '''
        changes:dict
        for changes in list_of_changes:
            self.set_a_new_training(changes)