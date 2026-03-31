"""inference/detector.py

Unified anomaly detector wrapper for the supported inference models:
  - "conv": PyTorch ConvAutoencoder on log-mel spectrograms
  - "mfcc": Keras autoencoder on normalized MFCC tensors
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


class AnomalyDetector:
    """Wrap a trained anomaly model and expose a common prediction API."""

    LABEL_NORMAL = "NORMAL"
    LABEL_ANOMALY = "ANOMALY"

    def __init__(
        self,
        model_path: str,
        threshold: float,
        model_type: str = "conv",
        mean_path: str | None = None,
        std_path: str | None = None,
        device: str | None = None,
    ):
        self.model_type = (model_type or "conv").lower()
        self.model_path = self._resolve_path(model_path)
        self.threshold = float(threshold)
        self.mean_path = mean_path
        self.std_path = std_path
        self.device_name = device

        self.model: Any = None
        self.device: Any = None
        self.criterion: Any = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.mfcc_input_shape: tuple[int | None, ...] | None = None
        self._torch = None

        if self.model_type == "conv":
            self._init_conv_backend(device=device)
        elif self.model_type == "mfcc":
            self.mean, self.std = self._load_mfcc_stats()
        else:
            raise ValueError(
                f"Unsupported inference.model_type='{self.model_type}'. "
                "Expected 'conv' or 'mfcc'."
            )

        self.model = self._load_model()

    def predict_from_features(self, features: np.ndarray) -> tuple[float, str]:
        """Score a model-ready input tensor and map it to a label."""
        if self.model_type == "conv":
            error = self._predict_conv(features)
        else:
            error = self._predict_mfcc(features)

        label = self.LABEL_ANOMALY if error >= self.threshold else self.LABEL_NORMAL
        return error, label

    def predict_from_mel(self, mel: np.ndarray) -> tuple[float, str]:
        if self.model_type != "conv":
            raise ValueError("predict_from_mel() is only valid when model_type='conv'.")
        return self.predict_from_features(mel)

    def predict_from_mfcc(self, mfcc_input: np.ndarray) -> tuple[float, str]:
        if self.model_type != "mfcc":
            raise ValueError("predict_from_mfcc() is only valid when model_type='mfcc'.")
        return self.predict_from_features(mfcc_input)

    def predict_from_file(self, path: str, cfg: dict) -> tuple[float, str]:
        """Load a WAV file, build the right features, and run inference."""
        if self.model_type == "conv":
            from audio.features import wav_to_mel

            features = wav_to_mel(path, cfg)
            return self.predict_from_mel(features)

        from audio.features_mfcc import wav_to_mfcc_input

        features = wav_to_mfcc_input(
            path,
            cfg,
            mean=self.mean,
            std=self.std,
            target_frames=self._get_mfcc_target_frames(),
        )
        return self.predict_from_mfcc(features)

    def _init_conv_backend(self, device: str | None) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for inference.model_type='conv'."
            ) from exc

        self._torch = torch
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def _predict_conv(self, mel: np.ndarray) -> float:
        x = self._torch.tensor(mel, dtype=self._torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            x_hat = self.model(x)
            return float(self.criterion(x_hat, x).item())

    def _predict_mfcc(self, mfcc_input: np.ndarray) -> float:
        x = np.asarray(mfcc_input, dtype=np.float32)
        self._validate_mfcc_input_shape(x)
        x_hat = self.model.predict(x, verbose=0)
        return float(np.mean((x - x_hat) ** 2, axis=(1, 2, 3))[0])

    def _load_model(self) -> Any:
        if self.model_type == "conv":
            return self._load_conv_model()
        return self._load_mfcc_model()

    def _load_conv_model(self) -> Any:
        from models.conv_autoencoder import ConvAutoencoder

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Conv model file not found: {self.model_path}")

        ckpt = self._torch.load(self.model_path, map_location=self.device)
        latent_dim = ckpt.get("latent_dim")

        model = ConvAutoencoder(latent_dim=latent_dim).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"Model loaded  | type=conv | latent_dim={latent_dim} | device={self.device}")
        return model

    def _load_mfcc_model(self) -> Any:
        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ImportError(
                "TensorFlow/Keras is required for inference.model_type='mfcc'."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MFCC model file not found: {self.model_path}")

        model = load_model(self.model_path, compile=False)
        self.mfcc_input_shape = tuple(model.input_shape)
        print(f"Model loaded  | type=mfcc | input_shape={model.input_shape}")
        return model

    def _load_mfcc_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.mean_path or not self.std_path:
            raise ValueError(
                "inference.mean_path and inference.std_path are required when "
                "model_type='mfcc'."
            )

        mean_path = self._resolve_path(self.mean_path)
        std_path = self._resolve_path(self.std_path)

        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"MFCC mean file not found: {mean_path}")
        if not os.path.exists(std_path):
            raise FileNotFoundError(f"MFCC std file not found: {std_path}")

        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        return mean, std

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)

    def _get_mfcc_target_frames(self) -> int | None:
        if not self.mfcc_input_shape or len(self.mfcc_input_shape) < 3:
            return None
        frames = self.mfcc_input_shape[2]
        if frames is None:
            return None
        return int(frames)

    def _validate_mfcc_input_shape(self, x: np.ndarray) -> None:
        if not self.mfcc_input_shape or len(self.mfcc_input_shape) != 4:
            return

        expected_n_mfcc = self.mfcc_input_shape[1]
        expected_frames = self.mfcc_input_shape[2]
        expected_channels = self.mfcc_input_shape[3]

        if x.ndim != 4:
            raise ValueError(
                f"MFCC input must have shape (N, n_mfcc, T, C), got {x.shape}"
            )

        if expected_n_mfcc is not None and x.shape[1] != int(expected_n_mfcc):
            raise ValueError(
                f"MFCC input mismatch: model expects n_mfcc={expected_n_mfcc}, got {x.shape[1]}"
            )
        if expected_frames is not None and x.shape[2] != int(expected_frames):
            raise ValueError(
                f"MFCC input mismatch: model expects frames={expected_frames}, got {x.shape[2]}"
            )
        if expected_channels is not None and x.shape[3] != int(expected_channels):
            raise ValueError(
                f"MFCC input mismatch: model expects channels={expected_channels}, got {x.shape[3]}"
            )
