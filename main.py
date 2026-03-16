#!/usr/bin/env python3
"""main.py â€“ Unified entry point for the anomaly-detection pipeline.

Two config files are loaded and merged automatically:
  config.yaml         â€“ DSP / model / training params  (original, unchanged)
  runtime_config.yaml â€“ inference path, recording, LCD â€¦

Usage examples
--------------
# Live mode â€“ record -> filter -> infer (don't save audio to disk)
python main.py --mode live

# Live mode + save valid audio
python main.py --mode live --save-dir /home/tesis/Audios

# Batch mode â€“ process a folder of WAVs
python main.py --mode batch --audio-dir /home/tesis/Audios/2026-03-06

# Override threshold or model path without editing any file
python main.py --mode batch --audio-dir /data --threshold 0.15 --model-path /models/new.pth

# Use non-default config files
python main.py --mode live --config /path/to/config.yaml --runtime-config /path/to/runtime_config.yaml
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config_loader      import load_config
from audio.recorder     import AudioRecorder
from audio.preprocess   import should_discard
from audio.features     import wav_to_mel
from inference.detector import AnomalyDetector
from display.lcd        import LCDDisplay


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time / batch audio anomaly detection."
    )
    p.add_argument(
        "--mode",
        choices=["live", "batch"],
        required=True,
        help="'live'  â€“ continuous recording loop.  "
             "'batch' â€“ process all WAVs in --audio-dir.",
    )

    # â”€â”€ config files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml  (default: config.yaml next to main.py).",
    )
    p.add_argument(
        "--runtime-config",
        default=None,
        metavar="PATH",
        help="Path to runtime_config.yaml  "
             "(default: runtime_config.yaml next to main.py).",
    )

    # â”€â”€ inference overrides â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Override inference.model_path from runtime_config.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override inference.threshold from runtime_config.",
    )

    # â”€â”€ live-mode options â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="[live] Save valid recorded audio here.  "
             "Omit to run inference without saving to disk.",
    )

    # â”€â”€ batch-mode options â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument(
        "--audio-dir",
        default=None,
        metavar="DIR",
        help="[batch] Folder containing WAV files to process.",
    )

    return p.parse_args()


# â”€â”€ pipeline helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _print_result(filename: str, error: float, label: str) -> None:
    icon = "X" if label == "ANOMALY" else "OK"
    print(f"  [{icon}]  {label:<8}  error={error:.6f}  file={filename}")


def _run_inference(mel, detector: AnomalyDetector, lcd: LCDDisplay, filename: str):
    error, label = detector.predict_from_mel(mel)
    _print_result(filename, error, label)
    lcd.show_error(error, label)
    return error, label


# â”€â”€ live mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_live(cfg: dict, detector: AnomalyDetector, lcd: LCDDisplay, save_dir):
    """Continuous recording loop.

    Record -> quality-gate -> (optional save) -> feature extraction -> inference.
    When save_dir is None a temp file is used for the quality gate and deleted
    immediately after, so nothing is written to disk.
    """
    import tempfile
    import soundfile as sf

    recorder = AudioRecorder(cfg)

    print("\n=== Live mode started ===")
    print(f"  threshold  = {detector.threshold}")
    print(f"  model      = {detector.model_path}")
    print(f"  save audio = {save_dir or 'no'}")
    print("  Press Ctrl+C to stop.\n")

    lcd.clear()

    while True:
        # 1. disk-space guard (only when saving)
        if save_dir and not recorder.has_enough_space(save_dir):
            print("Stopping: not enough disk space.")
            break

        # 2. record
        print("Recording...")
        audio = recorder.record()

        # 3. write WAV (permanent or temp)
        tmp_path = None
        if save_dir:
            day_dir  = recorder.today_subdir(save_dir)
            wav_path = recorder.save(audio, day_dir)
        else:
            tmp      = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            sf.write(tmp_path, audio, cfg["audio"]["sample_rate"], subtype="PCM_16")
            wav_path = tmp_path

        # 4. quality gate
        discard, reason = should_discard(wav_path, cfg)

        if discard:
            print(f"Discarded - {reason}")
            lcd.show(reason)
            if tmp_path:
                os.unlink(tmp_path)
            time.sleep(recorder.sleep_seconds)
            continue

        lcd.show("valid")

        # 5. feature extraction -> inference
        mel = wav_to_mel(wav_path, cfg)

        if tmp_path:
            os.unlink(tmp_path)
            tmp_path = None

        _run_inference(mel, detector, lcd, os.path.basename(wav_path))

        time.sleep(recorder.sleep_seconds)


# â”€â”€ batch mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_batch(cfg: dict, detector: AnomalyDetector, lcd: LCDDisplay, audio_dir: str):
    """Process every WAV file in audio_dir.

    Quality-gate -> feature extraction -> inference for each file.
    """
    if not os.path.isdir(audio_dir):
        print(f"ERROR: --audio-dir does not exist: {audio_dir}")
        sys.exit(1)

    wav_files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))

    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return

    print(f"\n=== Batch mode  |  {len(wav_files)} files  |  dir={audio_dir} ===")
    print(f"  threshold = {detector.threshold}")
    print(f"  model     = {detector.model_path}\n")

    results = {"NORMAL": 0, "ANOMALY": 0, "discarded": 0}

    for filename in wav_files:
        path = os.path.join(audio_dir, filename)
        print(f"File: {filename}")

        discard, reason = should_discard(path, cfg)
        if discard:
            print(f"   Discarded - {reason}")
            results["discarded"] += 1
            continue

        try:
            mel          = wav_to_mel(path, cfg)
            error, label = _run_inference(mel, detector, lcd, filename)
            results[label] += 1
        except Exception as exc:
            print(f"   Error: {exc}")

    total = len(wav_files)
    print(f"\n-- Summary --")
    print(f"  Total files : {total}")
    print(f"  NORMAL      : {results['NORMAL']}")
    print(f"  ANOMALY     : {results['ANOMALY']}")
    print(f"  Discarded   : {results['discarded']}")


# â”€â”€ entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    args = parse_args()

    # build optional CLI overrides
    overrides: dict = {}
    if args.model_path:
        overrides.setdefault("inference", {})["model_path"] = args.model_path
    if args.threshold is not None:
        overrides.setdefault("inference", {})["threshold"] = args.threshold

    # load + merge both configs
    cfg = load_config(
        base_path=args.config,
        runtime_path=args.runtime_config,
        overrides=overrides or None,
    )

    # shared components
    inf_cfg  = cfg["inference"]
    detector = AnomalyDetector(
        model_path=inf_cfg["model_path"],
        threshold=float(inf_cfg["threshold"]),
    )
    lcd = LCDDisplay(cfg)

    # dispatch
    if args.mode == "live":
        try:
            run_live(cfg, detector, lcd, save_dir=args.save_dir)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            lcd.clear()

    elif args.mode == "batch":
        if not args.audio_dir:
            print("ERROR: --audio-dir is required in batch mode.")
            sys.exit(1)
        run_batch(cfg, detector, lcd, audio_dir=args.audio_dir)


if __name__ == "__main__":
    main()
