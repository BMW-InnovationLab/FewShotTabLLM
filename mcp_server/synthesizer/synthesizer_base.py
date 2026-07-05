"""
Base class for all synthesizer wrappers.
"""

import os
import pandas as pd


class Synthesizer:
    """base model"""

    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.synthesizer = None

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.synthesizer.save(filepath=self.model_path)

    def load_model(self):
        raise NotImplementedError

    def export_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Write *data* to CSV and return the absolute path."""
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, filename)
        data.to_csv(path, index=False)
        return os.path.abspath(path)