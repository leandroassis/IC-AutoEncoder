from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, CSVLogger, TensorBoard
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.keras.losses import MSE, MeanAbsoluteError
from modules.misc import LSSIM, ssim_metric, psnrb_metric, LossLinearCombination, AdversarialLoss
from modules.DataMod import DataSet
from modules.TrainingManager import KerasTrainingManager
from os import environ
import tensorboard

environ["CUDA_VISIBLE_DEVICES"]="1"


def generator_training(self: KerasTrainingManager) -> None:

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
        tensorboard = TensorBoard(self.file_path)

        self.callbacks.append(csv_logger)
        self.callbacks.append(tensorboard)
    
        last_epoch = self._get_last_epoch_()

        self.fit_kwargs['epochs'] += 1 + last_epoch

        neural_net.fit(x = x_train, y = y_train,
                       validation_data = (x_test, y_test),
                       initial_epoch = last_epoch + 1,
                       callbacks = self.callbacks,
                       **self.fit_kwargs)

        neural_net.save(filepath = self.model_save_pathname)






manager2 = KerasTrainingManager(
    "AutoEncoder-1.0-64x64.json",
    optimizer = Adam,
    optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
    loss = LossLinearCombination,
    loss_kwargs = {'losses' : [AdversarialLoss, MeanAbsoluteError], 'weights' : [1,1], 'bias_vector' : [0,0]},
    compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
    
    fit_kwargs = {'batch_size' : 20, 'epochs' : 20, 'verbose':1, 'validation_split':0, 'shuffle':True, 
    'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
    'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

    metrics = [ssim_metric],

    training_function = generator_training,

    dataset = DataSet().load_discriminator_training_set(),
    new = True
)

manager2.start_training()


manager1 = KerasTrainingManager(
    "AutoEncoder-1.0-64x64.json",
    optimizer = Adam,
    optimizer_kwargs = {'learning_rate' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-7, 'amsgrad' : False},
    loss = LossLinearCombination,
    loss_kwargs = {'losses' : [AdversarialLoss(adversarial_model= 'Discriminator-AutoEncoder-1.0-64x64.json'), MeanAbsoluteError], 'weights' : [1,1], 'bias_vector' : [0,0]},
    compile_kwargs = {'loss_weights' : None, 'weighted_metrics' : None, 'run_eagerly' : None, 'steps_per_execution' : None},
    
    fit_kwargs = {'batch_size' : 20, 'epochs' : 20, 'verbose':1, 'validation_split':0, 'shuffle':True, 
    'class_weight':None, 'sample_weight':None, 'steps_per_epoch':None, 'validation_steps':None, 
    'validation_batch_size':None, 'validation_freq':1, 'max_queue_size':10, 'workers':1, 'use_multiprocessing':False},

    metrics = [ssim_metric],

    training_function = generator_training,

    dataset = DataSet().load_rafael_cifar_10_noise_data(),
    new = True
)

manager1.start_training()
