import ast
from pandas import DataFrame, read_csv


class DataHandle ():

    def __init__(self, 
                 trainings_file_path = "/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/Logs/Trainings_data.csv",
                 models_file_path = "/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/Logs/models_data.csv",
                 models_columns: list = [
                    "model_name",
                    "total_params",
                    "params_distribution",
                    "total_layers",
                    "total_treinable_layers",
                    "total_paths",
                    "total_kernel_regularizers",
                    "total_bias_regularizers",
                    "total_activity_regularizers",
                    "kernel_regularizers_distribution",
                    "bias_regularizers_distribution",
                    "activity_regularizers_distribution",
                    "kernel_regularizers_L1_distribution",
                    "bias_regularizers_L1_distribution",
                    "activity_regularizers_L1_distribution",
                    "kernel_regularizers_L2_distribution",
                    "bias_regularizers_L2_distribution",
                    "activity_regularizers_L2_distribution",
                    "initializer_count",
                    "dropout_count",
                    "dropout_distribution",
                    "batch_normalization_count",
                    "batch_normalization_distribution"
                 ],
                 training_columns: list = [
                    "training_idx",
                    "model_name",
                    "optimizer_args",
                    "loss_args",
                    "data_set_info",
                    "best_results",
                    "date_time",
                    "training_time_seconds"
                    "training_time",
                    "n_epochs"
                 ]) -> None:
        
        self.trainings_file_path = trainings_file_path
        self.models_file_path = models_file_path

        self.training_dataframe = read_csv(trainings_file_path, converters= dict.fromkeys(training_columns, self.literal_converter))
        self.models_dataframe = read_csv(models_file_path, converters= dict.fromkeys(models_columns, self.literal_converter))

    def literal_converter (self, data) -> any:
        try: 
            return ast.literal_eval(str(data))
        except SyntaxError: # probably str
            return data
    
    
