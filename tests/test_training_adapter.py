from unittest import mock
import unittest

from mlctlsriracha.plugins.azureml.train import AzureMlTrain
from mlctlsriracha.plugins.gcpvertex.train import GcpVertexTrain
from mlctlsriracha.plugins.awssagemaker.train import AwsSageMakerTrain


class TestTrainingAdapter(unittest.TestCase):

    # @mock.patch('mlctlsriracha.plugins.awssagemaker.train.Path.mkdir', return_value=[Tru])
    def test_awssagemaker_plugin(self):
        with mock.patch('mlctlsriracha.plugins.awssagemaker.train.Path.mkdir', autospec=True) as mock_mkdir:
            mock_mkdir.return_value = True
            plugin = AwsSageMakerTrain()
            
        response = plugin.finish()
        self.assertTrue(response)
    
    
    def test_awssagemaker_data(self):
         with mock.patch('mlctlsriracha.plugins.awssagemaker.train.Path.mkdir', autospec=True) as mock_mkdir:
            with mock.patch('mlctlsriracha.plugins.awssagemaker.train.glob.glob', autospec=True) as mock_glob:
                mock_mkdir.return_value = True
                mock_glob.return_value = ['./static/data/data.csv']
                plugin = AwsSageMakerTrain()
               
                response = plugin.input_as_dataframe()
                # print(response.size)
                self.assertTrue(response.size == 15)

    def test_awssagemaker_folder(self):
         with mock.patch('mlctlsriracha.plugins.awssagemaker.train.Path.mkdir', autospec=True) as mock_mkdir:
                mock_mkdir.return_value = True
                plugin = AwsSageMakerTrain()
               
                response = plugin.log_artifact('model.pkl')
                print(response)
                self.assertTrue(response == '/opt/ml/model/model.pkl')

    def test_azureml_plugin(self):
        with mock.patch('mlctlsriracha.plugins.azureml.train.Path.mkdir', autospec=True) as mock_mkdir:
            mock_mkdir.return_value = True
            plugin = AzureMlTrain()
        
        response = plugin.finish()
        self.assertTrue(response)

    # def test_gcpvertex_plugin(self):
    #     plugin = AwsSageMakerTrain()
    #     response = plugin.finish()
    #     self.assertTrue(response)