"""
CTGAN generation wrapper.

Loads a previously-saved model and samples new rows.
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer

from .synthesizer_base import Synthesizer


class CTGANGenerator(Synthesizer):
    """Load a trained CTGAN model and sample synthetic rows."""

    def load_model(self) -> None:
        self.synthesizer = CTGANSynthesizer.load(filepath=self.model_path)

    def generate(self, num_rows: int, filename: str) -> pd.DataFrame:
        """Sample *num_rows* rows, export to CSV, and return the DataFrame."""
        synthetic_data = self.synthesizer.sample(num_rows=num_rows)
        self.export_csv(synthetic_data, filename)
        return synthetic_data


