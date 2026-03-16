"""audio/features.py

Converts a WAV file (or a float32 waveform array) into a log-mel
spectrogram tensor ready for inference.

All DSP parameters are read from the 'audio', 'windowing', and
'spectrogram' sections of config.yaml, so the feature extraction
pipeline stays in sync with training.
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf


# ── helpers ──────────────────────────────────────────────────────────────────

def _expected_frames(
    window_sec: float,
    sr: int,
    n_fft: int,
    hop_length: int,
    center: bool = False,
) -> int:
    n_samples = int(round(window_sec * sr))
    if center:
        n_samples += n_fft
    if n_samples < n_fft:
        return 1
    return 1 + (n_samples - n_fft) // hop_length


def _rms_normalize(
    y: np.ndarray,
    target_rms: float = 0.1,
    eps: float = 1e-8,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize audio to a fixed RMS level (per window).
    This makes the pipeline more robust to mic gain / distance changes.

    y: float32 mono waveform
    target_rms: desired RMS amplitude (typical values: 0.05 to 0.2)
    """
    if y.size == 0:
        return y.astype(np.float32)
    rms  = float(np.sqrt(np.mean(y ** 2)))
    gain = target_rms / (rms + eps)
    y2   = (y * gain).astype(np.float32)
    return np.clip(y2, -1.0, 1.0) if clip else y2


def _normalize_audio(y: np.ndarray, mode: str, target_rms: float = 0.1) -> np.ndarray:
    """
    mode: 'none' or 'rms'
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return y
    if mode == "rms":
        return _rms_normalize(y, target_rms=target_rms)
    raise ValueError(f"Unknown audio normalization mode: '{mode}'")


# ── public API ────────────────────────────────────────────────────────────────

def wav_to_mel(path: str, cfg: dict) -> np.ndarray:
    """Load a WAV and return a log-mel spectrogram (float32, shape [n_mels, T]).

    Args:
        path: Path to a mono/stereo WAV file.
        cfg:  Full config dict.

    Returns:
        2-D float32 array of shape (n_mels, T_target).
    """
    sr         = cfg["audio"]["sample_rate"]
    window_sec = cfg["windowing"]["window_sec"]
    norm_mode  = cfg["audio"].get("normalization", "none")
    target_rms = float(cfg["audio"].get("target_rms"))

    sp         = cfg["spectrogram"]
    n_mels     = int(sp["n_mels"])
    n_fft      = int(sp["n_fft"])
    hop_length = int(sp["hop_length"])
    fmin       = sp.get("fmin")
    fmax       = sp.get("fmax")
    log_eps    = float(sp.get("log_eps", 1e-6))
    center     = bool(sp.get("center"))

    y, _ = librosa.load(path, sr=sr)
    y    = _normalize_audio(y, mode=norm_mode, target_rms=target_rms)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=center,
    )
    logS = np.log(S + log_eps).astype(np.float32)

    # Enforce the exact time dimension used during training
    t_target = _expected_frames(window_sec, sr, n_fft, hop_length, center=center)
    _, T = logS.shape
    if T > t_target:
        logS = logS[:, :t_target]
    elif T < t_target:
        logS = np.pad(logS, ((0, 0), (0, t_target - T)), mode="constant")

    return logS  # (n_mels, T_target)


def array_to_mel(audio_f32: np.ndarray, cfg: dict) -> np.ndarray:
    """Same as wav_to_mel but accepts a waveform array directly
    (useful for the live mode where we never write to disk).

    Args:
        audio_f32: float32 mono waveform at cfg['audio']['sample_rate'].
        cfg:       Full config dict.

    Returns:
        2-D float32 array of shape (n_mels, T_target).
    """
    import tempfile, os

    # Write to a temp WAV so we can reuse wav_to_mel cleanly
    sr = cfg["audio"]["sample_rate"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio_f32, sr, subtype="PCM_16")
        return wav_to_mel(tmp_path, cfg)
    finally:
        os.unlink(tmp_path)
