from sriracha.interfaces.Predict import PredictInterface

class AwsSageMakerPredict(PredictInterface):

    def __init__(self, profile=None):
        print('Selected Azure ML profile')

    def model(self, filename):
        """
        The path to model artifacts.

        Loads from the environmental variable that mlctl passes to Azure ML

        Arguments:
            filename (str): The name of the file which will be written back to S3

        Returns:
            path (str): The absolute path to the model output directory
        """
        model_uri = os.environ.get('AZUREML_MODEL_DIR')
        return os.path.join(model_uri, filename)

    def endpoint_metadata():
        return {
            'container_port': os.getenv('AIP_HTTP_PORT'),
            'model_id': os.getenv('AIP_DEPLOYED_MODEL_ID')
        }