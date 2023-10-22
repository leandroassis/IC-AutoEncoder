from tensorflow.keras.callbacks import Callback

'''

filepath=self.filepath+train_name+'{epoch:02d}-{val_loss:.2f}.hdf5',
                                save_weights_only=True,
                                monitor=self.metric,
                                save_freq = self.save_freq
                                verbose=self.verbose)
'''

class WeightCheckpoint(Callback):
    def __init__(self, epoch_offset : int, monitor : str, save_path : str, json_dict : dict, save_best=True, direction='max', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.epoch_offset = epoch_offset
        self.path = save_path
        self.save_best_only = save_best
        self.direction = direction
        self.verbose = verbose
        
        self.last = float('-inf') if self.direction == 'max' else float('inf')
        self.log = json_dict
    
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("ModelCheckpoint requires %s available!" % self.monitor, RuntimeWarning)
        
        if self.save_best_only == False:
            self.write_output("Salvando pesos da época %02d." %(epoch+epoch_offset))
        else:
            if direction == "max" and current > self.last:
                self.write_output("Época %02d: A métrica %s cresceu de %.6f para %.6f. Salvando pesos."  %(epoch+epoch_offset) %self.monitor %self.last %current)
            elif direction == "min" and current < self.last:
                self.write_output("Época %02d: A métrica %s desceu de %.6f para %.6f. Salvando pesos."  %(epoch+epoch_offset) %self.monitor %self.last %current)
        self.model.save_weights(self.path+"%02d.hdf5" %(epoch+epoch_offset), overwrite=True, save_format='h5')
        
        self.last = current
        self.log['current_epoch'] = epoch+epoch_offset
        
    def write_output(self, string):
        if self.verbose > 0:
            print(string)

class EarlyStopByPercentage(Callback):
    def __init__(self, monitor : str, percentage : float, num_epochs : int, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.percentage = percentage
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.last = []

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        
        if len(self.last) != self.num_epochs:
            self.last.append(current)
            return
        
        if self.last[0]/current < 1 + (percentage/100):
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        else:
            self.last = [current]