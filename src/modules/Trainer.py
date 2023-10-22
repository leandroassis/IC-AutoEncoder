# trainer com mlflow pra logar (opcional) + cross validation + treinar até não ter melhoria

from modules.DataMod import DataSet
from src.modules.CustomCallbacks import EarlyStopByPercentage, WeightCheckpoint
from sklearn.model_selection import StratifiedKFold
import json
from os import makedirs, path

from numpy.random import seed

class NetTrainer():

    def __init__(self, model,
                 dataset : DataSet,
                 epochs : int,
                 batch_size : int,
                 tracking_metric : str,
                 logs_path : str,
                 grow_percentage : float = 0.2,
                 grow_time : int = 7,
                 no_folds : int = 10,
                 save_freq = "epoch",
                 hyperparams = None,
                 verbose : int = 1,
                 fix_seed : int = 344343):
        
        self.model = model
        self.hps = hyperparams
        self.dataset = dataset

        self.num_epochs = epochs
        self.batch_size = batch_size
        
        self.verbose = verbose
        self.metric = tracking_metric
        self.percentage = grow_percentage
        self.grow_epochs = grow_time
        self.save_freq = save_freq
        
        self.filepath = logs_path if logs_path[-1] == '/' else logs_path+'/'
        
        self.no_folds = no_folds
        
        if not path.exists(logs_path):
            makedirs(logs_path)
            
        try:
            with open(self.filepath+"status.json", "r") as json_file:
                self.json = json.load(json_file)
        except IOError:
                self.json = {'current_fold':1, 'current_epoch':0, 'status':'RUNNING'}
                
        seed(fix_seed)
    def start_training(self):
        
        if self.json['status'] == 'COMPLETED':
            print('Treino finalizado. Foram executados %02d folds com %02 épocas em cada.' %self.no_folds %self.num_epochs)
            return
        
        last_fold = self.__get_last_fold()
        last_epoch = self.__get_last_epoch()
        
        self.skf = StratifiedKFold(n_splits=no_folds)
        self.stop = EarlyStopByPercentage(monitor=self.metric, percentage=self.percentage, num_epochs=self.grow_epochs, verbose=self.verbose)
        self.checkpoint = WeightCheckpoint(last_epoch, self.metric, fold_path, json_dict = self.json, save_best=True, direction='max', verbose=1)
        
        current_fold = 1
        for train_index, test_index in self.skf.split(self.dataset.x_train, self.dataset.y_train):            
            if current_fold < last_fold:
                current_fold+=1
                continue
                
            if self.verbose > 0 and current_fold > 1:
                    print("Continuando treino a partir do fold: %02d" %current_fold)
         
            fold_path = logs_path+ "fold_%02d/" % current_fold
            if not path.exists(fold_path):
                makedirs(fold_path)

            if self.verbose > 0 and last_epoch > 0:
                print("Continuando treino a partir da época: %02d" %last_epoch)

            # recreates the model
            self.model = self.model() if self.hps == None else self.model(self.hps)
            
            if last_epoch != 0:
                self.model.load_weights(fold_path+'%02d.hdf5' % last_epoch)
            
            self.history = self.model.fit(self.dataset.x_train[train_index], self.dataset.y_train[train_index], epochs=(self.num_epochs-last_epoch),
                           batch_size=self.batch_size, validation_data=(self.dataset.x_train[test_index], self.dataset.y_train[test_index]),
                           callbacks=[self.checkpoint, self.stop])
            current_fold+=1
            self.json['current_fold'] = current_fold

        self.json['status'] = 'COMPLETED'

    def __get_last_epoch(self) -> int:
        return self.json['current_epoch']
    
    def __get_last_fold(self) -> int:
        return self.json['current_fold']
        
        
        