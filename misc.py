from tensorflow._api.v2.image import ssim
from datetime import datetime as dt
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.losses import Loss, Reduction


class LSSIM (Loss):

    def __init__(self, name = "LSSIM", reduction = Reduction.AUTO, max_val = 255, filter_size=9, filter_sigma=1.5, k1=0.01, k2=0.03) -> None:
        
        super(LSSIM, self).__init__(name = name, reduction = reduction)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

    
    def call (self,y_true,y_pred):
        return 1-ssim(y_true, y_pred, max_val = self.max_val,
                      filter_size = self.filter_size,
                      filter_sigma = self.filter_sigma,
                      k1 = self.k1,
                      k2 = self.k2)



def get_current_time_and_data ():
    '''
        Retorna o tempo em horas:minutos, e data em ano-mes-dia (str)
    '''
    current_datetime = dt.now()
    time = current_datetime.time()
    date = current_datetime.date()
    date_str = str(date)
    time_str = time.strftime('%X')[:5]
    return date_str, time_str


def get_last_epoch (csv_pathname):
    """
        Retorna a ultima época treinada de um checkpoint. \n
        obs: retorna -1 quando nenhum treino foi realizado para o treino inciar na época 0.
    """
    try:
        file = open(csv_pathname, 'r')
    except FileNotFoundError:
        return -1
    lines = file.read().splitlines()
    if (len(lines) == 0):
        return -1
    file.close()
    last_line = lines[-1]
    last_epoch = int(last_line.split(';')[0])
    return last_epoch



