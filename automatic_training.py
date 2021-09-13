import pickle
from os.path import isdir
from os import makedirs
from tensorflow.keras.models import model_from_json, Model
from pandas import DataFrame, read_csv
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.lib.io.file_io import file_exists
from misc import get_current_time_and_data, get_last_epoch
from DataMod import DataSet

from os.path import dirname


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
        dataframe_name:str = "Results_DataFrame"
        self.dataframe_pathname:str = self.sub_dir_0 + dataframe_name

        # Dataframe params
        self.dataframe_columns:list = ["training idx",
                                        "date", 
                                        "model name",
                                        "dataset",
                                        "dataset params",
                                        "regularizer",
                                        "model total params",
                                        "model total layers",
                                        "optimizer",
                                        "optimizer args",
                                        "loss", 
                                        "loss args",
                                        "compile args",
                                        "fit args",
                                        "best training loss",
                                        "best training epoch",
                                        "best validation loss",
                                        "best validation epoch",
                                        "last epoch training loss",
                                        "last epoch validation",
                                        "last epoch"]
        
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
        print("Atualizando atributos dependentes.")
        self.date, self.time = get_current_time_and_data()
        self.image_name = '#' + str(self.training_idx) + '|' + self.date + "|" + self.time + "|" + "|epoch=" + str(self.last_epoch + self.number_of_ephocs) + ".png"
        
        # diretorios base
        self.sub_dir_0 = "Relatorios-Dados-etc/Resultados/"
        self.sub_dir_1 = self.dataset_name + '/'
        self.sub_dir_2 = self.model_name.replace('.json', '') + '/'
        self.sub_dir_3 = str(self.training_idx) + '/'
        self.data_path = self.sub_dir_0 + self.sub_dir_1 + self.sub_dir_2 + self.sub_dir_3

        # CSV_Logger
        csv_name:str = 'csv-#' + str(self.training_idx) + ".log"
        self.csv_pathname:str = self.data_path + csv_name

        # last epoch
        self.last_epoch:int = get_last_epoch(self.csv_pathname)

        # Model training name
        self.model_save_pathname:str = self.data_path + '#' + str(self.training_idx) + '-checkp'
    
    def change_atributes (self, kw_att_and_val:dict) -> None:
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
        print("Mudando atributos selecionados")

        def recursive_update (dictionary:dict, Kw):
            for key, value in kw_att_and_val.items():
                # all this is to check if the atribute of type 'dict' will be replaced or have some key values changed.
                if type(value) == dict:
                    if key[0] == '*': # replace the dict
                        dictionary[key[1:]] = value
                    else:
                        recursive_update(dictionary[key], value) # change key values.
            else:
                dictionary[key] = value

        recursive_update(self.__dict__, kw_att_and_val)
        self._update_dependent_atributes_()


    def change_training_idx (self) -> None:
        self.training_idx += 1
        self._update_dependent_atributes_()

    def show (self):
        
        print('\n Training data: \n')
        training_data:list = ['training_idx', 'model_name', 'dataset_name',
        'number_of_ephocs', 'fit_Kwargs', 'last_epoch', 'loss_kwargs',
        'loss', 'optimizer', 'optimizer_kwargs', 'compile_kwargs']

        for key in training_data:
            print(
                key + ' =', self.__dict__[key]
            )

        print('\n Dir names: \n')
        dir_data:list = ['model_save_pathname', 'csv_pathname', 'data_path', 'dataframe_pathname']
        for key in dir_data:
            print(
                key + ' =', self.__dict__[key]
            )

        print()

# ======================================================================================================================

# ======================================================================================================================

class Auto_Training ():
    """
        ## Função:

        A classe implementa o treino e a geração automatica de logs e resultados usando o keras e outras bibliotecas.
    """

    def __init__(self, load_state = True) -> None:
        self.version:str = '0.0.0'
        self.state_pathname:str = "Auto_Training_state/" + "state-v:" + self.version + ".ats"
        self.state:Training_State = Training_State()

        if load_state:
            self.load_state()

    def save_state(self) -> None:
        '''
            ## Função :

            Salva o estado atual do objeto.
        '''
        print("Saving the actual state.")
        self._check_if_dirs_exists_()
        with open(self.state_pathname, 'wb') as file:
            pickle.dump(self.state, file, pickle.HIGHEST_PROTOCOL)
            file.close()
        

    def load_state(self) -> None:
        '''
            ## Função :

            Carrega o estado armazenado no arquivo
        '''
        print("Loading the actual state.")
        if file_exists(self.state_pathname):

            with open(self.state_pathname, 'rb') as file:
                preveous_obj:Training_State = pickle.load(file)
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
        print("checking if the dirs exists")

        if not isdir( dirname(self.state.csv_pathname) ):
            makedirs( dirname(self.state.csv_pathname) )

        if not isdir( dirname(self.state.data_path) ):
            makedirs( dirname(self.state.data_path) )

        if not isdir( dirname(self.state_pathname) ):
            makedirs( dirname(self.state_pathname) )

        if not isdir( dirname(self.state.dataframe_pathname) ):
            makedirs ( dirname(self.state.dataframe_pathname) )

        if not isdir( dirname(self.state.model_save_pathname) ):
            makedirs ( dirname(self.state.model_save_pathname) )



        
        
    def _load_model_ (self) -> Model:
        '''
            ## Função:

            Carrega o modelo de rede neural definido no atributo `model_name`, 
            junto dos pesos armazenados no checkpoint definido no atibuto `checkpoint_name`
        '''

        # Tentando carregar o modelo
        print("Trying to load a preveous state of training.")
        if file_exists(self.state.model_save_pathname):
            nNet:Model = load_model(self.state.model_save_pathname, custom_objects = {self.state.loss().name : self.state.loss}, compile = True)
            print('Preveous state loaded.')
            return nNet

        print("Loading just the model.")
        try:
            json_file = open(self.state.models_dir + self.state.model_name, "r")
        except FileNotFoundError:
            raise Exception("Fail atempt to load the model " + self.state.model_name + " at " + self.state.models_dir)

        json_readed = json_file.read()
        nNet:Model = model_from_json(json_readed)
        json_file.close()

        return nNet
    


    def get_csv_training_history (self) -> DataFrame:
        """

        """
        dataframe = read_csv(self.state.csv_pathname, sep=';')
        return dataframe

    def get_best_results(self) -> dict:
        """
        
        """
        dataframe:DataFrame = self.get_csv_training_history ()

        best_training_loss = dataframe["loss"].min()
        best_training_epoch = [line.epoch for line in dataframe.itertuples() if line.loss == best_training_loss][0]

        best_val_loss = dataframe["val_loss"].min()
        best_val_epoch = [line.epoch for line in dataframe.itertuples() if line.val_loss == best_val_loss][0]

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

        dataset:DataSet = DataSet()
        dataset.load_by_name(self.state.dataset_name)
        
        x_train = dataset.x_train
        x_test = dataset.x_test
        y_train = dataset.y_train
        y_test = dataset.y_test
        

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
        self.save_state()



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