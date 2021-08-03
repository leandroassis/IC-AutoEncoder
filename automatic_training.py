import pickle

from os.path import isdir, dirname
from os import makedirs
from typing import Any
from numpy import str0
from numpy.lib.npyio import load
from tensorflow.keras.models import model_from_json


from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from misc import get_current_time_and_data, get_last_epoch

from DataMod import DataSet


class Training_State ():

    def set_state(self, model_name:str, training_idx:int, fit_Kwargs:dict, dataset:DataSet, 
    compile_kwargs:dict, loss_class:Loss, loss_kwargs:dict, optimizer_class:Optimizer, optimizer_kwargs:dict) -> None:
        """
            ### Função:

            Define um novo estado do zero.
        """

        # Parametros de treino
        self.training_idx:int = training_idx
        self.model_name:str = model_name
        self.fit_Kwargs:dict = fit_Kwargs

        #dataset
        self.dataset:DataSet = dataset

        #data e tempo
        (self.date, self.time) = get_current_time_and_data()

        ## Caminhos e nomes de arquivos

        self.data_pathname:str = "Relatorios-Dados-etc/Resultados"
        self.sub_dir_1 = self.dataset.name + '/'
        self.sub_dir_2 = self.model_name.replace('.json', '')
        self.sub_dir_3 = '-#' + str(training_idx)

        #checkpoints
        checkpoints_dir_path:str = "checkpoints/"
        checkpoint_dir_name:str = self.model_name.replace('.json', '-#' + str(training_idx)) + '-' + dataset.name + '/'
        checkpoint_name:str = '#' + str(self.training_idx) + '-checkp'
        self.checkpoint_pathname:str = checkpoints_dir_path + checkpoint_dir_name + checkpoint_name

        #dados do treino
        training_data_dir_path:str = "Relatorios-Dados-etc/Parametros e dados de Resultados/"
        training_data_dir_name:str = self.model_name.replace('.json', '|') + self.dataset.name + "/"
        csv_name:str = '#' + str(self.training_idx) + ".log"
        self.csv_pathname:str = training_data_dir_path + training_data_dir_name + csv_name 

        #imagens do treino
        images_dir_path = "Relatorios-Dados-etc/Imagens de resultados/"
        image_dir_name =  dataset.name + '/' + self.model_name.replace('.json', '') + '/'
        image_name = '#' + str(self.training_idx) + '|' + self.date + "|" + self.time + "|" + "|epoch=" + str(self.actual_epoch) + ".png"
        self.image_directory_pathname = self.images_dir + self.image_dir_name + self.image_name

        self.models_dir:str = "nNet_models/"
        
        #Parametros de treino
        self.last_epoch:int = get_last_epoch(self.csv_pathname)
        self.actual_epoch:int = self.last_epoch + self.fit_Kwargs['ephocs']
        self.loss:Loss = loss_class
        self.loss_kwargs:dict = loss_kwargs
        
        #parametros do compiler
        self.optimizer:Optimizer = optimizer_class
        self.optimizer_kwargs:dict = optimizer_kwargs

    
    def save_state(self, path_name:str) -> None:
        '''
            ### Função :

            Salva o estado atual do objeto.
        '''
        with open(path_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            file.close()
        

    def load_state(self, path_name:str) -> None:
        '''
            ### Função :

            Carrega o estado armazenado no arquivo
        '''
        with open(path_name, 'rb') as file:
            preveous_obj:Training_State = pickle.load(file, pickle.HIGHEST_PROTOCOL)
            self.__dict__ = preveous_obj.__dict__ # fazendo os atributos do objeto atual igual ao objeto salvo no arquivo.
            file.close()


    def change_atributes (self, Kw_att_and_val:dict) -> None:
        """
            ### Função:

            Muda determinados atributos especificadoes na forma de keyword e valor.

            ### Exemplo:
            
            Caso queira mudar o nome do modelo a ser usado e do otimizador escreva:

            >>> Kw_att_and_val = {'model_name': 'nome', 'optimizer': optimizer_class}
            # note que 'model_name' e 'optimizer' são atributos de Auto_training
        """
        for key, value in Kw_att_and_val.items():
            self.__dict__[key] = value


# ======================================================================================================================

# ======================================================================================================================

class Auto_Training ():
    """
        ### Função:

        A classe implementa o treino e a geração automatica de logs e resultados usando o keras e outras bibliotecas.
    """

    def __init__(self) -> None:
        self.state_pathname:str
        self.state:Training_State
        self.state.load_state(self.state_pathname)
        pass

        
    def _update_date_time (self) -> None:
        '''
            Atualiza atributos que dependem do tempo.
        '''
        self.date, self.time = get_current_time_and_data()
        self.image_name = '#' + str(self.training_idx) + '|' + self.date + "|" + self.time + "|" + "|epoch=" + str(self.actual_epoch) + ".png"

   
    def _check_if_dirs_exist (self) -> None:
        """
            ### Função:

            Verifica se todos os diretórios que compoem um caminho existem.
            E no caso de não existirem, o método cria esses diretórios.

            ### Exemplo: 
                * `Relatorios-Dados-etc/Imagens de resultados`

            São dois diretórios, ambos serão criados caso ja não existam na pasta
            onde o programa é executado.
        """

        # Checkpoints

        checkpoint_path = dirname(self.state.checkpoint_pathname)
        if not isdir(checkpoint_path):
            makedirs(checkpoint_path)

        
        



    def _load_model_and_checkpoit (self) -> Any:
        '''
            ### Função:

            Carrega o modelo de rede neural definido no atributo `model_name`, 
            junto dos pesos armazenados no checkpoint definido no atibuto `checkpoint_name`
        '''

        # Tentando carregar o modelo
        try:
            json_file = open(self.models_dir + self.model_name, "r")
        except FileNotFoundError:
            raise("Fail atempt to load the model " + self.model_name + " at " + self.models_dir)

        nNet = json_file.read()

        json_file.close()

        # Tentando carregar o checkpoin
        




    
    def training(self) -> None:
        """
            Metodo que inicia o treino
        """
        x_train = self.dataset.x_train
        x_test = self.dataset.x_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test
        
        
        
