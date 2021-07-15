from mlsriracha.interfaces.metadata import MetadataInterface

class MlFlowMetadata(MetadataInterface):

    def __init__(self, run_name):
        print('Selected MLFlow profile')
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag('mlsriracha', '0.0.1')

    def log_param(params):
        for key, value in params.items():
            print('Params: ', key, ': ', value)
            mlflow.log_param(key, value)

    def log_metric(params):
        mlflow.log_metrics(params) 

    # conversion to MLFlow required artifact type for model inference
    class ModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model
            
        def predict(self, context, model_input):
            return self.model.predict_proba(model_input)[:,1]

    def log_artifact(object, type='model'):
        mlflow_model = ModelWrapper(object)
        mlflow.pyfunc.log_model(artifact_path=run_name, python_model=mlflow_model)
        pass

    def finish():
        pass