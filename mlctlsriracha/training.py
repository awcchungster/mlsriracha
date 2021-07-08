from mlctlsriracha.plugins.azureml.train import AzureMlTrain
from mlctlsriracha.plugins.gcpvertex.train import GcpVertexTrain

class TrainingAdapter:
    def __init__(self,
        provider: str):
        self.provider_name = provider.lower()

        print('This is a training job')
        if provider.lower() == 'azureml':
            print('Using Azure ML as a provider')
            self.provider_obj = AzureMlTrain()
        else if provider.lower() == 'gcpvertex':
            print('Using GCP Vertex as a provider')
            self.provider_obj = GcpVertexTrain()
        
            
    def input_as_dataframe(self, channel: str):
        return self.provider_obj.input_as_dataframe(channel)

    def log_artifact(self, filename: str):
        return self.provider_obj.log_artifact(filename)

    def finish():
        return self.provider_obj.finish()