#!/usr/bin/env python3
"""Unified entry point for the anomaly-detection pipeline.

Two config files are loaded and merged automatically:
  config.yaml         - DSP / model / training params
  runtime_config.yaml - inference path, recording, LCD, AWS IoT

Usage examples
--------------
# Live mode - record -> filter -> infer (don't save audio to disk)
python main.py --mode live

# Live mode + save valid audio
python main.py --mode live --save-dir /home/tesis/Audios

# Live mode + publish inference results to AWS IoT
python main.py --mode live --send-to-aws

# Batch mode - process a folder of WAVs
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

from config_loader import load_config
from audio.recorder import AudioRecorder
from audio.preprocess import should_discard
from inference.detector import AnomalyDetector
from display.lcd import LCDDisplay
from raspi_publish import AWSIoTPublisher


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time / batch audio anomaly detection."
    )
    parser.add_argument(
        "--mode",
        choices=["live", "batch"],
        required=True,
        help="'live' runs the continuous recorder. 'batch' processes WAV files.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: file next to main.py).",
    )
    parser.add_argument(
        "--runtime-config",
        default=None,
        metavar="PATH",
        help="Path to runtime_config.yaml (default: file next to main.py).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Override inference.model_path from runtime_config.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["conv", "mfcc"],
        help="Override inference.model_type from runtime_config.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override inference.threshold from runtime_config.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="[live] Save valid recorded audio here.",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        metavar="DIR",
        help="[batch] Folder containing WAV files to process.",
    )

    parser.add_argument(
        "--send-to-aws",
        dest="send_to_aws",
        action="store_true",
        help="Publish inference results to AWS IoT.",
    )
    parser.set_defaults(send_to_aws=False)

    return parser.parse_args()


# ── pipeline helpers ──────────────────────────────────────────────────────────

def _print_result(filename: str, error: float, label: str) -> None:
    icon = "X" if label == "ANOMALY" else "OK"
    print(f"  [{icon}]  {label:<8}  error={error:.6f}  file={filename}")


def _publish_result(
    publisher: AWSIoTPublisher,
    error: float,
    label: str,
) -> None:
    if not publisher.enabled:
        return

    payload = {
        "deviceId": publisher.device_id,
        "timestamp": int(time.time()), #  Unix timestamp, the number of seconds since 1970-01-01 00:00:00 UTC
        "result": label,
        "error": float(error),
        "modelRun": publisher.model_run,
    }

    try:
        publisher.publish(payload)
    except Exception as exc:
        print(f"Warning: failed to publish to AWS IoT: {exc}")


def _run_inference(
    wav_path: str,
    cfg: dict,
    detector: AnomalyDetector,
    lcd: LCDDisplay,
    filename: str,
    publisher: AWSIoTPublisher,
):
    error, label = detector.predict_from_file(wav_path, cfg)
    _print_result(filename, error, label)
    lcd.show_error(error, label)
    _publish_result(publisher, error, label)
    return error, label


def run_live(
    cfg: dict,
    detector: AnomalyDetector,
    lcd: LCDDisplay,
    save_dir,
    publisher: AWSIoTPublisher,
):
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
    print(f"  model type = {detector.model_type}")
    print(f"  model      = {detector.model_path}")
    print(f"  save audio = {save_dir or 'no'}")
    print(f"  send aws   = {'yes' if publisher.enabled else 'no'}")
    if publisher.enabled and publisher.backup_enabled:
        print(f"  aws backup = {publisher.backup_path}")
    print("  Press Ctrl+C to stop.\n")

    lcd.clear()
    publisher.flush_pending()

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
            day_dir = recorder.today_subdir(save_dir)
            wav_path = recorder.save(audio, day_dir)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
        # 5. run inference

        try:
            _run_inference(
                wav_path,
                cfg,
                detector,
                lcd,
                os.path.basename(wav_path),
                publisher,
            )
        finally:
            if tmp_path:
                os.unlink(tmp_path)
                tmp_path = None

        time.sleep(recorder.sleep_seconds)


# ── batch mode ────────────────────────────────────────────────────────────────
def run_batch(
    cfg: dict,
    detector: AnomalyDetector,
    lcd: LCDDisplay,
    audio_dir: str,
    publisher: AWSIoTPublisher,
):
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
    print(f"  model type = {detector.model_type}")
    print(f"  model     = {detector.model_path}")
    print(f"  send aws  = {'yes' if publisher.enabled else 'no'}\n")
    publisher.flush_pending()

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
            _, label = _run_inference(
                path,
                cfg,
                detector,
                lcd,
                filename,
                publisher,
            )
            results[label] += 1
        except Exception as exc:
            print(f"   Error: {exc}")

    total = len(wav_files)
    print("\n-- Summary --")
    print(f"  Total files : {total}")
    print(f"  NORMAL      : {results['NORMAL']}")
    print(f"  ANOMALY     : {results['ANOMALY']}")
    print(f"  Discarded   : {results['discarded']}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Build optional CLI overrides.
    overrides: dict = {}
    if args.model_path:
        overrides.setdefault("inference", {})["model_path"] = args.model_path
    if args.model_type:
        overrides.setdefault("inference", {})["model_type"] = args.model_type
    if args.threshold is not None:
        overrides.setdefault("inference", {})["threshold"] = args.threshold
    overrides.setdefault("aws_iot", {})["enabled"] = args.send_to_aws

    # Load + merge both configs.
    cfg = load_config(
        base_path=args.config,
        runtime_path=args.runtime_config,
        overrides=overrides or None,
    )

    # Shared components.
    inf_cfg = cfg["inference"]
    detector = AnomalyDetector(
        model_path=inf_cfg["model_path"],
        threshold=float(inf_cfg["threshold"]),
        model_type=inf_cfg.get("model_type", "conv"),
        mean_path=inf_cfg.get("mean_path"),
        std_path=inf_cfg.get("std_path"),
    )
    lcd = LCDDisplay(cfg)
    publisher = AWSIoTPublisher(cfg.get("aws_iot"))

    try:
        # Dispatch.
        if args.mode == "live":
            try:
                run_live(cfg, detector, lcd, save_dir=args.save_dir, publisher=publisher)
            except KeyboardInterrupt:
                print("\nStopped by user.")
                lcd.clear()
        elif args.mode == "batch":
            if not args.audio_dir:
                print("ERROR: --audio-dir is required in batch mode.")
                sys.exit(1)
            run_batch(
                cfg,
                detector,
                lcd,
                audio_dir=args.audio_dir,
                publisher=publisher,
            )
    finally:
        publisher.disconnect()


if __name__ == "__main__":
    main()
