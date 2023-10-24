# trainer com mlflow pra logar (opcional) + cross validation + treinar até não ter melhoria

from modules.DataMod import DataSet
from tensorflow.keras.callbacks import BackupAndRestore, CSVLogger
from src.modules.CustomCallbacks import EarlyStopByPercentage
from sklearn.model_selection import KFold
import json
from os import makedirs, path

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
        self.history = {}
        
        if not path.exists(logs_path):
            makedirs(logs_path)
            
        try:
            with open(self.filepath+"status.json", "r") as json_file:
                self.json = json.load(json_file)
        except IOError:
                self.json = {'path':self.filepath+"status.json", 'current_fold':1, 'current_epoch':0, 'status':'RUNNING'}
                
        self.seed = fix_seed
    def start_training(self):
        
        if self.json['status'] == 'COMPLETED':
            print('Treino finalizado. Foram executados %02d folds com %02 épocas em cada.' %(self.no_folds, self.num_epochs))
            return
        
        last_fold = self.__get_last_fold()
        
        self.stop = EarlyStopByPercentage(monitor=self.metric, percentage=self.percentage, num_epochs=self.grow_epochs, verbose=self.verbose)
        
        if self.no_folds == 1:
            self.checkpoint = BackupAndRestore(backup_dir=self.filepath+'fit_backup', save_freq="epoch",  delete_checkpoint=False)
            self.logger = CSVLogger(filename=self.filepath+'training.log', separator=';')

            # recreates the model
            self.model = self.model() if self.hps == None else self.model(self.hps)
            
            self.history['fold01'] = self.model.fit(self.dataset.x_train, self.dataset.y_train, epochs=self.num_epochs,
                           batch_size=self.batch_size, validation_data=(self.dataset.x_test, self.dataset.y_test),
                           callbacks=[self.checkpoint, self.stop, self.logger])
            return
        
        self.kf = KFold(n_splits=self.no_folds, shuffle=True, random_state=self.seed)
        current_fold = 1
        for train_index, test_index in self.kf.split(self.dataset.x_train, self.dataset.y_train):            
            if current_fold < last_fold:
                current_fold+=1
                continue
                
            if self.verbose > 0 and current_fold > 1:
                    print("Continuando treino a partir do fold: %02d" %current_fold)
         
            fold_path = self.filepath+ "fold_%02d/" % current_fold
            if not path.exists(fold_path):
                makedirs(fold_path)
            
            self.checkpoint = BackupAndRestore(backup_dir=fold_path+'fit_backup', save_freq="epoch")
            self.logger = CSVLogger(filename=fold_path+'training.log', separator=';')

            # recreates the model
            self.model = self.model() if self.hps == None else self.model(self.hps)
            
            self.history['fold%02d' %current_fold] = self.model.fit(self.dataset.x_train[train_index], self.dataset.y_train[train_index], epochs=self.num_epochs,
                           batch_size=self.batch_size, validation_data=(self.dataset.x_train[test_index], self.dataset.y_train[test_index]),
                           callbacks=[self.checkpoint, self.stop, self.logger])
            current_fold+=1
            self.json['current_fold'] = current_fold
            self.__update_json()

        self.json['status'] = 'COMPLETED'

    def __get_last_epoch(self) -> int:
        return self.json['current_epoch']
    
    def __get_last_fold(self) -> int:
        return self.json['current_fold']
    
    def __update_json(self):
        # Serializing json
        json_object = json.dumps(self.json, indent=4)

        # Writing to sample.json
        with open(self.json['path'], "w") as outfile:
            outfile.write(json_object)
        
        
        