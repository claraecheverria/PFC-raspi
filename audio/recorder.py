"""audio/recorder.py

Handles microphone capture, resampling, and optional saving to disk.
All settings are read from the 'recording' section of config.yaml.
"""

import os
import wave
import shutil
import datetime

import numpy as np
import librosa
import sounddevice as sd

try:
    import alsaaudio
    _HAS_ALSA = True
except ImportError:
    _HAS_ALSA = False


class AudioRecorder:
    """Records fixed-length audio segments from a USB microphone.

    Args:
        cfg: Full config dict (as returned by config_loader.load_config).
    """

    def __init__(self, cfg: dict):
        rec = cfg["recording"]
        self.card_index        = rec["card_index"]
        self.mic_volume        = int(rec["mic_volume"])
        self.samplerate        = int(rec["samplerate"])
        self.target_samplerate = int(rec["target_samplerate"])
        self.channels          = int(rec["channels"])
        self.duration          = float(rec["duration"])
        self.sleep_seconds     = float(rec["sleep_seconds"])
        self.threshold_gb      = int(rec["threshold_gb"])
        self.label             = rec["label"]

    # ── public ───────────────────────────────────────────────

    def has_enough_space(self, base_path: str) -> bool:
        """Return True if *base_path* has >= threshold_gb free."""
        _, _, free = shutil.disk_usage(base_path)
        free_gb = free // (1024 ** 3)
        print(f"💾 Free disk space: {free_gb} GB")
        return free_gb >= self.threshold_gb

    def record(self) -> np.ndarray:
        """Capture one segment and return a float32 mono waveform at
        *target_samplerate*.  Does NOT write anything to disk.
        """
        self._set_mic_volume()
        sd.default.device = self.card_index

        raw = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
        )
        sd.wait()

        audio_f32 = raw.squeeze().astype(np.float32) / 32768.0
        if self.target_samplerate != self.samplerate:
            audio_f32 = librosa.resample(
                y=audio_f32,
                orig_sr=self.samplerate,
                target_sr=self.target_samplerate,
            )
        return audio_f32

    def save(self, audio_f32: np.ndarray, save_dir: str) -> str:
        """Write *audio_f32* as a 16-bit WAV in *save_dir*.

        The filename includes a timestamp and the configured label.
        Returns the full path of the saved file.
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath  = os.path.join(save_dir, f"{self.label}_{timestamp}.wav")

        audio_i16 = np.clip(audio_f32 * 32768, -32768, 32767).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.target_samplerate)
            wf.writeframes(audio_i16.tobytes())

        print(f"💾 Saved: {filepath}")
        return filepath

    def today_subdir(self, base_path: str) -> str:
        """Return (and create) base_path/YYYY-MM-DD/."""
        today   = datetime.date.today().strftime("%Y-%m-%d")
        day_dir = os.path.join(base_path, today)
        os.makedirs(day_dir, exist_ok=True)
        return day_dir

    # ── private ──────────────────────────────────────────────

    def _set_mic_volume(self) -> None:
        if not _HAS_ALSA:
            return
        try:
            m = alsaaudio.Mixer(control="Mic", cardindex=2)
            m.setvolume(self.mic_volume)
            print(f"🎚  Mic volume: {m.getvolume()}")
        except Exception as exc:
            print(f"⚠  Could not set mic volume: {exc}")
