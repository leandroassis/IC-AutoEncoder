import numpy as np

def train_progress_is_low (epoch_results:list, metric_name:str, percentual_treashold = 1, average_window = 3, window_distance = 3):

    number_of_epochs = epoch_results.__len__() 

    if number_of_epochs < average_window + window_distance:
        return False
    
    last_results = epoch_results[metric_name][number_of_epochs - average_window : ]
    prev_results = epoch_results[metric_name][number_of_epochs - average_window - window_distance : number_of_epochs - average_window ]

    if (last_results/prev_results - 1)*100 < percentual_treashold and max(last_results) < max(prev_results):
        return True

    return False