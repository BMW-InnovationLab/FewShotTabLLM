"""
CTGAN training wrapper.

All CTGAN hyper-parameters live here so the service layer can
forward them as a flat ``**kwargs`` dict.
"""

from typing import List, Optional, Union

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

from .synthesizer_base import Synthesizer


class CTGANTrainer(Synthesizer):
    """Train a CTGAN model and persist it to disk."""

    def init_synthesizer(
        self,
        metadata: SingleTableMetadata,
        *,
        # ── data-processing knobs ────────────────────────────
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        locales: Optional[Union[List[str], str]] = None,
        # ── architecture ─────────────────────────────────────
        embedding_dim: int = 128,
        generator_dim: List[int] = [256, 256],
        discriminator_dim: List[int] = [256, 256],
        # ── optimiser ────────────────────────────────────────
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        # ── training ─────────────────────────────────────────
        batch_size: int = 500,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        verbose: bool = False,
        epochs: int = 300,
        pac: int = 10,
        cuda: Union[bool, str] = True,
    ) -> None:
        """CTGANSynthesizer``"""

        self.synthesizer = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales or ["en_US"],
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
        )

    def train(self, data: pd.DataFrame) -> None:
        """Fit the synthesizer on *data* and persist the model."""
        self.synthesizer.fit(data)
        self.save_model()

