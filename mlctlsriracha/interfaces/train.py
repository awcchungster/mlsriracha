
import pandas as pd
class TrainInterface:
    def init(self):
        """Load the model."""
        pass

    def input_as_dataframe(self, channel) -> pd:
        """Return the input data for that channel as a Panda dataframe """
        pass

    def log_artifact():
        """Returns the directory that the training job should output artifacts to"""
        pass

    def artifact_upload():
        """Upload all the artifacts saved to the URI provided by the provider"""
        pass