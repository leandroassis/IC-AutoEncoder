
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from modules.TrainingManager import KerasTrainingManager
from tensorflow.keras.models import Model

def generator_training(self: KerasTrainingManager, epochs = None) -> None:

        if epochs != None:
            self.fit_kwargs['epochs'] = epochs


        x_train = self.dataset.x_train
        x_test = self.dataset.x_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test
        

        neural_net: Model = self._get_model()    

        if self.optimizer_kwargs:
            optimizer = self.optimizer(**self.optimizer_kwargs)
        else:
            optimizer = self.optimizer()
        
        if self.loss_kwargs:
            loss = self.loss(**self.loss_kwargs)
        else:
            loss = self.loss()
            
            
        neural_net.compile(optimizer = optimizer,
                           loss = loss,
                           metrics = self.metrics,
                           **self.compile_kwargs)

        self.make_all_dirs()

        csv_logger = CSVLogger(filename = self.csv_pathname, separator = ';', append= True)
        tensorboard = TensorBoard(self.file_path)


        callbacks = [csv_logger, tensorboard]
        if self.callbacks:
            callbacks.append(self.callbacks)
    
        last_epoch = self._get_last_epoch_()

        self.fit_kwargs['epochs'] += 1 + last_epoch

        neural_net.fit(x = x_train, y = y_train,
                       validation_data = (x_test, y_test),
                       initial_epoch = last_epoch + 1,
                       callbacks = callbacks,
                       **self.fit_kwargs)

        neural_net.save(filepath = self.model_save_pathname)


def multiple_losses (self: KerasTrainingManager, epochs = None):

    
        if epochs != None:
            self.fit_kwargs['epochs'] = epochs


        x_train = self.dataset.x_train
        x_test = self.dataset.x_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test
        

        neural_net: Model = self._get_model()    

        if self.optimizer_kwargs:
            optimizer = self.optimizer(**self.optimizer_kwargs)
        else:
            optimizer = self.optimizer()

        losses = []
        for loss, kwarg in zip(self.loss, self.loss_kwargs):
            if kwarg:
                losses.append(loss(**kwarg))
            else:
                losses.append(loss())
            
            
        neural_net.compile(optimizer = optimizer,
                           loss = losses,
                           metrics = self.metrics,
                           **self.compile_kwargs)

        self.make_all_dirs()

        csv_logger = CSVLogger(filename = self.csv_pathname, separator = ';', append= True)
        tensorboard = TensorBoard(self.file_path)


        callbacks = [csv_logger, tensorboard]
        if self.callbacks:
            callbacks.append(self.callbacks)
    
        last_epoch = self._get_last_epoch_()

        self.fit_kwargs['epochs'] += 1 + last_epoch

        neural_net.fit(x = x_train, y = y_train,
                       validation_data = (x_test, y_test),
                       initial_epoch = last_epoch + 1,
                       callbacks = callbacks,
                       **self.fit_kwargs)

        neural_net.save(filepath = self.model_save_pathname)