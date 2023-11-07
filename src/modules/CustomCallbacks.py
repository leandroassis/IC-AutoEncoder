from tensorflow.keras.callbacks import Callback
import json

class WeightCheckpoint(Callback):
    def __init__(self, epoch_offset : int, monitor : str, save_path : str, json_dict : dict, direction='max', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.epoch_offset = epoch_offset
        self.path = save_path
        self.direction = direction
        self.verbose = verbose
        
        self.last = float('-inf') if self.direction == 'max' else float('inf')
        self.log = json_dict
    
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("ModelCheckpoint requires %s available!" % self.monitor, RuntimeWarning)
            
        current_epoch = epoch+self.epoch_offset+1
        
        if self.direction == "max" and current > self.last:
            self.write_output("\nÉpoca %02d: A métrica %s cresceu de %.6f para %.6f. Salvando pesos."  %(current_epoch, self.monitor, self.last, current))
        elif self.direction == "min" and current < self.last:
            self.write_output("\nÉpoca %02d: A métrica %s desceu de %.6f para %.6f. Salvando pesos."  %(current_epoch, self.monitor, self.last, current))
        self.model.save_weights(self.path+"%02d.hdf5" %current_epoch, overwrite=True, save_format='h5')
        
        self.last = current
        self.log['current_epoch'] = current_epoch
        
        json_object = json.dumps(self.log, indent=4)
        with open(self.log['path'], "w") as outfile:
            outfile.write(json_object)
        
    def write_output(self, string):
        if self.verbose > 0:
            print(string)

class EarlyStopByPercentage(Callback):
    def __init__(self, monitor : str, percentage : float, num_epochs : int, path : str, json_dict : dict, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.percentage = percentage
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.path = path
        self.log = json_dict
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
            with open(path, "w") as file:
                    file.write("Época %05d: encerrando pois não houve melhora nas últimas %02d épocas\n" %(epoch, self.num_epochs))
                    self.log['status'] = 'COMPLETED'
            self.model.stop_training = True
        else:
            self.log['status'] = 'RUNNING'
            self.last = [current]
            
        json_object = json.dumps(self.log, indent=4)
        with open(self.log['path'], "w") as outfile:
            outfile.write(json_object)