from sriracha.interfaces.Predict import PredictInterface

class GcpVertexPredict(PredictInterface):

    def __init__(self, profile=None):
        print('Selected GCP Vertex profile')

        Path('/opt/ml/model/').mkdir(parents=True, exist_ok=True)

        storage_client = storage.Client()

        bucket_name = os.getenv('AIP_STORAGE_URI')
        bucket = storage_client.bucket(bucket_name)
        # TODO GENERALIZE so it's not file specific
        
        artifact = f'{bucket_name}/model.pkl'
        print(f'GS Artifact location: {artifact}')
        blob = Blob.from_string(artifact, storage_client)
        blob.download_to_filename('/opt/ml/model/model.pkl')

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