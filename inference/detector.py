"""inference/detector.py

Loads a trained ConvAutoencoder checkpoint and runs anomaly detection.

Swapping the architecture only requires:
  1. Adding your model class under models/
  2. Changing the import and instantiation in AnomalyDetector._load_model().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class AnomalyDetector:
    """Wraps a trained autoencoder for anomaly scoring.

    Args:
        model_path: Path to the .pth checkpoint file.
        threshold:  Reconstruction-error threshold above which a sample is
                    labelled "ANOMALY".
        device:     "cuda", "cpu", or None (auto-detect).
    """

    LABEL_NORMAL  = "NORMAL"
    LABEL_ANOMALY = "ANOMALY"

    def __init__(
        self,
        model_path: str,
        threshold: float,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.threshold  = threshold
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.criterion  = nn.MSELoss(reduction="mean")
        self.model      = self._load_model()

    # ── public ───────────────────────────────────────────────

    def predict_from_mel(self, mel: np.ndarray) -> tuple[float, str]:
        """Run inference on a pre-computed mel spectrogram.

        Args:
            mel: float32 array of shape (n_mels, T).

        Returns:
            (reconstruction_error, label)  where label is "NORMAL" or "ANOMALY".
        """
        # [1, 1, n_mels, T]
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x_hat = self.model(x)
            error = self.criterion(x_hat, x).item()

        label = self.LABEL_ANOMALY if error >= self.threshold else self.LABEL_NORMAL
        return error, label

    def predict_from_file(self, path: str, cfg: dict) -> tuple[float, str]:
        """Convenience: load WAV → mel → inference.

        Args:
            path: Path to a WAV file.
            cfg:  Full config dict (passed through to audio.features.wav_to_mel).

        Returns:
            (reconstruction_error, label)
        """
        from audio.features import wav_to_mel
        mel = wav_to_mel(path, cfg)
        return self.predict_from_mel(mel)

    # ── private ──────────────────────────────────────────────

    def _load_model(self) -> nn.Module:
        # ── Import your architecture here ──────────────────
        # To swap architectures, replace this import and the
        # class instantiation below.
        from models.conv_autoencoder import ConvAutoencoder

        ckpt       = torch.load(self.model_path, map_location=self.device)
        latent_dim = ckpt.get("latent_dim")

        model = ConvAutoencoder(latent_dim=latent_dim).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"✅ Model loaded  | latent_dim={latent_dim} | device={self.device}")
        return model
