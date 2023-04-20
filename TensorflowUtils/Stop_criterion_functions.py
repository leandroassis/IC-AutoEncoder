def train_progress_is_low (epoch_mean_results: dict, 
                   metric_name: str,
                   best_metric_selector = max,
                   min_percentual_variation = 0.02/100,
                   observation_window_lenght = 7) -> bool: 

    if len(epoch_mean_results[metric_name]) < observation_window_lenght + 1:
        return False
    
    best_window_result = best_metric_selector(epoch_mean_results[metric_name][-observation_window_lenght:])

    best_result_outside_window = best_metric_selector(epoch_mean_results[metric_name][:-observation_window_lenght])

    stop_signal = (best_window_result - best_result_outside_window)/best_result_outside_window < min_percentual_variation

    return stop_signal