from mlctlsriracha.plugins.azureml.train import AzureMlTrain
from mlctlsriracha.plugins.gcpvertex.train import GcpVertexTrain
from mlctlsriracha.plugins.awssagemaker.train import AwsSageMakerTrain

class TrainingAdapter:
    def __init__(self,
        provider: str):
        self.provider_name = provider.lower()

        print('This is a training job')
        try: provider
        except NameError: 
            raise RuntimeError(f'{str} is not a valid provider')

        if provider.lower() == 'azureml':
            print('Using Azure ML as a provider')
            self.provider_obj = AzureMlTrain()
        elif provider.lower() == 'gcpvertex':
            print('Using GCP Vertex as a provider')
            self.provider_obj = GcpVertexTrain()
        elif provider.lower() == 'awssagemaker':
            print('Using AWS SageMaker as a provider')
            self.provider_obj = AwsSageMakerTrain()
        else:
            raise RuntimeError(f'{str} is not a valid provider')
        
            
    def input_as_dataframe(self, channel: str):
        return self.provider_obj.input_as_dataframe(channel=channel)

    def log_artifact(self, filename: str):
        return self.provider_obj.log_artifact(filename)

    def finish(self):
        return self.provider_obj.finish()