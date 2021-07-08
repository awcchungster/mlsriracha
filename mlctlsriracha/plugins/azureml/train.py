import os
import pandas as pd
from pathlib import Path

from mlctlsriracha.interfaces.train import TrainInterface

class AzureMlTrain(TrainInterface):

    def __init__(self, profile=None):
        print('Selected Azure ML profile')

    def input_as_dataframe(self, channel):
        """
        The path to input artifacts.

        In Azure, mlctl passes environment variables that map to the 
        train.yaml inputs that the user defines.

        Arguments:
            channel (str): The name of the channel which contains the given filename
            filename (str): The name of the file within a specific channel

        Returns:
            path (str): The absolute path to the specified channel file
        """

        # channel --> environmental variable names
        data_directories = {
            'training': "training-data",
            'validation': "validation-data",
            'testing': "testing-data"
        }

        if channel in data_directories.keys():
            azure_mount_file = os.environ.get(data_directories[channel])
            print(f'azure_mount_file={azure_mount_file}')
            data = pd.read_csv(azure_mount_file)
            return data

        else:
            print('Incorrect data channel type. Options are training, validation, and testing.')
            return null
    
    def log_artifact(self, filename):
        """
        The path to the output artifacts.

        Your algorithm should write all final model artifacts to this directory.
        Azure ML copies this data as a folder into the console as a Run output.

        Arguments:
            filename (str): The name of the file which will be written back to S3

        Returns:
            path (str): The absolute path to the model output directory
        """
        cwd = os.getcwd()
        Path('./outputs/model').mkdir(parents=True, exist_ok=True)
        return os.path.join(cwd, 'outputs', 'model', filename)

    def finish():
        pass