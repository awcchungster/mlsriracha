import os

class PredictInterface:
    def init(self):
        """Load the model."""
        pass

    def modelArtifactPath(self, filename=None) -> os.path:
        """Get the file directory and model path"""
        pass

    def extract_text(self, full_file_name: str) -> dict:
        """Extract text from the currently loaded file."""
        pass