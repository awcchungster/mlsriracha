from mlctlsriracha.plugins.azureml.train import AzureMlTrain
from mlctlsriracha.plugins.gcpvertex.predict import AzureMlTrain

class PredictAdapter:
    def __init__(self,
        provider: str):
        self.provider_name = provider.lower()

        print('This is a prediction job')
        if provider.lower() == 'azureml':
            print('Using Azure ML as a provider')
            self.provider_obj = AzureMlPredict()
        elif provider.lower() == 'azureml':
            print('Using Azure ML as a provider')
            self.provider_obj = GcpVertexPredict()
        
            
    def model(self, filename: str):
        return self.provider_obj.model(filename)

    