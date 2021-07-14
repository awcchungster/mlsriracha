import pandas as pd
import os
import glob
from pathlib import Path

from mlctlsriracha.interfaces.train import TrainInterface

class AwsSageMakerTrain(TrainInterface):

    def __init__(self):
        print('Selected AWS SageMaker ML profile')
        Path('/opt/ml/model').mkdir(parents=True, exist_ok=True)

    def input_as_dataframe(self, channel='training'):
        """
        The function returns a panda dataframe for the input channel artifacts.

        AWS SageMaker passes all values in FileMode from the S3 bucket into the 
        train, test, and validation "channels" when starting your container.
        This function takes in the channel and merges all CSV files per channel
        into a dataframe for reading.

        Arguments:
            channel (str): The name of the channel which contains the given filename

        Returns:
            path (str): The absolute path to the specified channel file
        """

        data_directories = {
            'training',
            'validation',
            'testing'
        }

        if channel in data_directories:

            csv_files = glob.glob(os.path.join(f'/opt/ml/input/data/{channel}/*.csv'))
            print(f'Files in {channel} directory: {csv_files}')
            # loop over the list of csv files
            fileBytes = []
            for f in csv_files:
        
                # read the csv file
                df = pd.read_csv(f)
                fileBytes.append(df)
            frame = pd.concat(fileBytes, axis=0, ignore_index=True)     
            return frame 
        else:
            raise ValueError('Incorrect data channel type. Options are training, validation, and testing.')
            return null
    
    def log_artifact(self, filename=''):
        """
        The path to the output artifacts.

        Your algorithm should write all final model artifacts to this directory.
        Amazon SageMaker copies this data as a single object in compressed tar
        format to the S3 location that you specified in the CreateTrainingJob
        request. If multiple containers in a single training job write to this
        directory they should ensure no file/directory names clash. Amazon SageMaker
        aggregates the result in a tar file and uploads to S3.

        Arguments:
            filename (str): The name of the file which will be written back to S3

        Returns:
            path (str): The absolute path to the model output directory
        """
        return os.path.join(os.sep, 'opt', 'ml', 'model', filename)

    def finish(self):
        return True