from pandas import DataFrame, read_csv
from tensorflow.python import training
from tensorflow.python.lib.io.file_io import file_exists


class CsvWriter():
    """
        This class defines how the data is saved in a file.
    """

    def __init__(self, file_name, training_idx) -> None:
        """
            Method that defines the name of the archive were data is saved etc

            receives: 
                    file_name (without extentions)
            return: 
                None
            raise: 
                None
        """
        self.training_idx = training_idx
        self.pathname = f"logs/{file_name}.csv"


    def write_data_to_table (self, columns_and_values: dict) -> None:
        """
            Obs: All the dict data has to be str or number.
        """
        new_data = DataFrame([columns_and_values])

        if file_exists(self.pathname):
            dataframe = read_csv(self.pathname, index_col=0)
        else:
            dataframe = DataFrame()
            

        if dataframe.empty:
            dataframe = dataframe.append(new_data, ignore_index=True)

        else:
            if not dataframe.iloc[-1]['training_idx'] == self.training_idx:
                dataframe = dataframe.append(new_data, ignore_index=True)
            else:
                dataframe = dataframe.drop(dataframe.last_valid_index(), axis=0)
                dataframe = dataframe.append(new_data, ignore_index=True)

        dataframe.to_csv(self.pathname)
