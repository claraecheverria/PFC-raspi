"""audio/features_mfcc.py

MFCC feature extraction helpers for the Keras MFCC autoencoder.

The core signal-processing steps mirror the notebook implementation:
  - librosa.load(..., sr=16000, mono=True)
  - RMS normalization
  - librosa.feature.mfcc(...)
  - pad / crop to a fixed number of time frames

Additional helpers prepare a single inference sample using the saved
train-set mean/std tensors and the input shape expected by ae 1.h5.
"""

from __future__ import annotations

import numpy as np
import librosa


def _rms_normalize_audio(
    y: np.ndarray,
    target_rms: float = 0.1,
    eps: float = 1e-8,
    clip: bool = True,
) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)

    rms = float(np.sqrt(np.mean(y ** 2)))
    gain = target_rms / (rms + eps)
    y2 = (y * gain).astype(np.float32)
    if clip:
        y2 = np.clip(y2, -1.0, 1.0).astype(np.float32)

    return y2


def _get_mfcc_cfg(cfg: dict) -> dict:
    mfcc_cfg = dict(cfg.get("mfcc", {}))
    return {
        "sr": int(cfg["audio"]["sample_rate"]),
        "n_mfcc": int(mfcc_cfg.get("n_mfcc")),
        "frame_ms": float(mfcc_cfg.get("frame_ms")),
        "hop_ms": float(mfcc_cfg.get("hop_ms")),
        "n_mels": int(mfcc_cfg.get("n_mels")),
        "target_frames": int(mfcc_cfg.get("target_frames")),
        "target_rms": float(mfcc_cfg.get("target_rms")),
    }


def _resolve_target_frames(cfg: dict, target_frames: int | None = None) -> int:
    if target_frames is not None:
        return int(target_frames)
    mfcc_cfg = _get_mfcc_cfg(cfg)
    return int(mfcc_cfg["target_frames"])


def load_audio(file_path: str, sr: int, target_rms: float = 0.1) -> np.ndarray:
    """Load mono audio and apply the RMS normalization used in training."""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return _rms_normalize_audio(audio, target_rms=target_rms)


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int,
    frame_ms: float,
    hop_ms: float,
    n_mels: int,
) -> np.ndarray:
    """Extract MFCCs with the same FFT / hop settings used in the notebook."""
    n_fft = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
    )
    return mfcc.astype(np.float32)


def pad_mfcc(mfcc: np.ndarray, target_frames: int = 64) -> np.ndarray:
    """Pad or crop an MFCC matrix to the fixed time dimension used by the model."""
    t_frames = mfcc.shape[1]

    if t_frames < target_frames:
        pad_width = target_frames - t_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    elif t_frames > target_frames:
        mfcc = mfcc[:, :target_frames]

    return mfcc.astype(np.float32)


def normalize_mfcc(
    mfcc: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize one sample like the notebook evaluation path.

    Input:
      mfcc shape: (n_mfcc, T)

    Output:
      normalized shape: (1, n_mfcc, T)
    """
    mfcc = np.asarray(mfcc, dtype=np.float32)
    if mfcc.ndim != 2:
        raise ValueError(f"Expected mfcc with shape (n_mfcc, T), got {mfcc.shape}")

    x = np.expand_dims(mfcc, axis=0)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((x - mean) / (std + eps)).astype(np.float32)


def reshape_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Convert a normalized batch from (N, n_mfcc, T) to (N, n_mfcc, T, 1)."""
    mfcc = np.asarray(mfcc, dtype=np.float32)
    if mfcc.ndim != 3:
        raise ValueError(
            f"Expected normalized MFCC batch with shape (N, n_mfcc, T), got {mfcc.shape}"
        )

    x = np.expand_dims(mfcc, axis=-1)
    return x.astype(np.float32)


def wav_to_mfcc(path: str, cfg: dict, target_frames: int | None = None) -> np.ndarray:
    """Load a WAV and return a padded/cropped MFCC matrix."""
    mfcc_cfg = _get_mfcc_cfg(cfg)
    audio = load_audio(
        file_path=path,
        sr=mfcc_cfg["sr"],
        target_rms=mfcc_cfg["target_rms"],
    )
    mfcc = extract_mfcc(
        audio=audio,
        sr=mfcc_cfg["sr"],
        n_mfcc=mfcc_cfg["n_mfcc"],
        frame_ms=mfcc_cfg["frame_ms"],
        hop_ms=mfcc_cfg["hop_ms"],
        n_mels=mfcc_cfg["n_mels"],
    )
    return pad_mfcc(mfcc, target_frames=_resolve_target_frames(cfg, target_frames))


def array_to_mfcc(
    audio_f32: np.ndarray,
    cfg: dict,
    target_frames: int | None = None,
) -> np.ndarray:
    """Extract MFCCs directly from a waveform array."""
    mfcc_cfg = _get_mfcc_cfg(cfg)
    audio = np.asarray(audio_f32, dtype=np.float32)
    audio = _rms_normalize_audio(audio, target_rms=mfcc_cfg["target_rms"])
    mfcc = extract_mfcc(
        audio=audio,
        sr=mfcc_cfg["sr"],
        n_mfcc=mfcc_cfg["n_mfcc"],
        frame_ms=mfcc_cfg["frame_ms"],
        hop_ms=mfcc_cfg["hop_ms"],
        n_mels=mfcc_cfg["n_mels"],
    )
    return pad_mfcc(mfcc, target_frames=_resolve_target_frames(cfg, target_frames))


def wav_to_mfcc_input(
    path: str,
    cfg: dict,
    mean: np.ndarray,
    std: np.ndarray,
    target_frames: int | None = None,
) -> np.ndarray:
    """Return a single normalized MFCC inference sample shaped for Keras."""
    mfcc = wav_to_mfcc(path, cfg, target_frames=target_frames)
    mfcc = normalize_mfcc(mfcc, mean=mean, std=std)
    return reshape_mfcc(mfcc)


def array_to_mfcc_input(
    audio_f32: np.ndarray,
    cfg: dict,
    mean: np.ndarray,
    std: np.ndarray,
    target_frames: int | None = None,
) -> np.ndarray:
    """Return a normalized MFCC inference sample from a waveform array."""
    mfcc = array_to_mfcc(audio_f32, cfg, target_frames=target_frames)
    mfcc = normalize_mfcc(mfcc, mean=mean, std=std)
    return reshape_mfcc(mfcc)
