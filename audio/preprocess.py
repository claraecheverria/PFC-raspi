"""audio/preprocess.py

Quality gate applied to every audio segment before inference.

Returns (discard: bool, reason: str | None) where reason is one of:
    "off machine"  – machine appears to be off
    "loud noise"   – sudden transient / external event
    None           – segment is valid
"""

from __future__ import annotations

import numpy as np
import librosa


# ── helpers ──────────────────────────────────────────────────────────────────

def _smooth_majority(mask: np.ndarray, window: int, frac: float = 0.6) -> np.ndarray:
    if window <= 1:
        return mask
    return (
        np.convolve(mask.astype(float), np.ones(window) / window, mode="same") > frac
    )


def _contiguous_segments(
    mask: np.ndarray, times: np.ndarray
) -> list[tuple[float, float]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segments, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((times[start], times[prev]))
        start = prev = i
    segments.append((times[start], times[prev]))
    return segments


# ── main function ─────────────────────────────────────────────────────────────

def should_discard(
    filepath: str,
    cfg: dict,
) -> tuple[bool, str | None]:
    """Analyse *filepath* and decide whether to discard it.

    Args:
        filepath: Path to a mono WAV file.
        cfg:      Full config dict.  Uses the 'preprocess' section.

    Returns:
        (discard, reason)
    """
    p  = cfg["preprocess"]
    sr = p["frame_length"]          # reuse SR from audio section if needed
    sr = cfg["audio"]["sample_rate"]

    frame_length = int(p["frame_length"])
    hop_length   = int(p["hop_length"])

    y, _ = librosa.load(filepath, sr=sr, mono=True)

    S    = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    mag  = np.abs(S) + 1e-12
    freq = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    times = librosa.frames_to_time(
        np.arange(mag.shape[1]), sr=sr, hop_length=hop_length
    )

    # ── "off machine" ────────────────────────────────────────
    band = (freq >= p["band_low"]) & (freq <= p["band_high"])
    band_rms  = np.sqrt(np.mean(mag[band, :] ** 2, axis=0) + 1e-12)
    band_dbfs = librosa.amplitude_to_db(band_rms, ref=1.0)

    off_mask = _smooth_majority(
        band_dbfs < p["band_off_threshold"],
        int(p["off_smooth_window"]),
    )
    for t0, t1 in _contiguous_segments(off_mask, times):
        if (t1 - t0) >= p["off_trigger_seconds"]:
            return True, "off machine"

    # ── "loud noise" ─────────────────────────────────────────
    rms_db = librosa.amplitude_to_db(
        librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        + 1e-12,
        ref=1.0,
    )
    loud_mask = _smooth_majority(
        rms_db > p["abs_loud_threshold_dbfs"],
        int(p["loud_smooth_window"]),
    )
    for t0, t1 in _contiguous_segments(loud_mask, times):
        if (t1 - t0) >= p["loud_trigger_seconds"]:
            return True, "loud noise"

    return False, None
