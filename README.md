# Raspberry Pi Audio Anomaly Detection

This repository contains an inference pipeline for detecting audio anomalies from a machine using a Raspberry Pi, a USB microphone, an optional I2C LCD, and optional AWS IoT publishing.

The pipeline can run in:

- `live` mode: record audio continuously, discard invalid segments, extract log-mel spectrograms, run the model, show the result on the LCD, and optionally publish it to AWS IoT
- `batch` mode: process every `.wav` file in a directory using the same preprocessing and inference steps

The repo already includes a trained convolutional autoencoder checkpoint at [models/model_best.pth](models/model_best.pth).

## What is in this repo

- [main.py](main.py): unified CLI entry point for live and batch inference
- [config.yaml](config.yaml): DSP, spectrogram, model, and training-related parameters
- [runtime_config.yaml](runtime_config.yaml): deployment/runtime settings for inference, recording, LCD, and AWS IoT
- [config_loader.py](config_loader.py): merges both config files and applies CLI overrides
- [audio/recorder.py](audio/recorder.py): microphone capture, resampling, and optional WAV saving
- [audio/preprocess.py](audio/preprocess.py): quality gate to reject `off machine` and `loud noise` segments
- [audio/features.py](audio/features.py): converts audio to the log-mel representation expected by the model
- [inference/detector.py](inference/detector.py): loads the checkpoint and computes the reconstruction error
- [display/lcd.py](display/lcd.py): optional HD44780 I2C LCD output
- [raspi_publish.py](raspi_publish.py): optional AWS IoT Core publisher

## How it works

### Live mode

1. Record one audio segment from the configured microphone
2. Save it either to a temporary file or to `--save-dir`
3. Run the quality gate
4. If valid, convert it to a log-mel spectrogram
5. Run the autoencoder and compute reconstruction error
6. Label the segment as `NORMAL` or `ANOMALY` using the configured threshold
7. Show the result on the LCD and optionally publish it to AWS IoT

### Batch mode

1. Read every `.wav` file in `--audio-dir`
2. Apply the same quality gate
3. Extract features
4. Run inference
5. Print a summary of normal, anomaly, and discarded files

## Requirements

Python `3.10+` is recommended.

Core Python packages:

- `numpy`
- `PyYAML`
- `librosa`
- `soundfile`
- `sounddevice`
- `torch`

Optional Python packages:

- `smbus2` for the LCD
- `AWSIoTPythonSDK` for AWS IoT publishing
- `pyalsaaudio` if you want the script to set microphone volume through ALSA

Typical install:

Create a virtual environment first. A virtual environment is an isolated Python environment for this project, so the packages you install here do not affect your system-wide Python or other projects.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pyyaml librosa soundfile sounddevice torch smbus2 AWSIoTPythonSDK pyalsaaudio
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pyyaml librosa soundfile sounddevice torch smbus2 AWSIoTPythonSDK pyalsaaudio
```

Notes:

- `sounddevice` usually needs PortAudio available on the system.
- `soundfile` may need `libsndfile` installed on Linux.
- On Raspberry Pi, the LCD also depends on I2C being enabled in the OS.
- If `smbus2` is unavailable, the LCD becomes a no-op.
- AWS IoT publishing is skipped unless it is explicitly enabled.
- If AWS IoT is enabled but the Pi loses internet, payloads are queued on disk and retried automatically.
- If `pyalsaaudio` is unavailable, ALSA mic-volume control is skipped.

## Configuration

The app merges two YAML files at startup:

- [config.yaml](config.yaml): audio DSP, spectrogram, model, threshold/training-related values
- [runtime_config.yaml](runtime_config.yaml): deployment paths and runtime settings

`runtime_config.yaml` overrides values from `config.yaml` when keys collide, and CLI arguments can override both.

Important sections:

- `audio`: sample rate and normalization used by feature extraction
- `windowing`: expected window length used to fix the spectrogram width
- `spectrogram`: mel-spectrogram parameters
- `inference`: model checkpoint path and anomaly threshold
- `recording`: live capture settings such as device, duration, sleep interval, and disk space guard
- `preprocess`: quality-gate thresholds for `off machine` and `loud noise`
- `lcd`: LCD bus/address settings
- `aws_iot`: endpoint, topic, certificates, and publish behavior

For resilient live deployments, `aws_iot.backup_enabled` stores failed publishes in `aws_iot.backup_path` as JSON Lines so the Pi can keep recording and inferencing while offline.

## Usage

### Show CLI help

```bash
python main.py --help
```

### Live inference

Record continuously, do not save audio to disk:

```bash
python main.py --mode live
```

Record continuously and save valid audio files:

```bash
python main.py --mode live --save-dir /home/tesis/Audios
```

Record continuously and publish predictions to AWS IoT:

```bash
python main.py --mode live --send-to-aws
```

### Batch inference

Process every WAV file in a folder:

```bash
python main.py --mode batch --audio-dir /home/tesis/Audios/2026-03-06
```

Override the threshold or checkpoint path from the command line:

```bash
python main.py --mode batch --audio-dir /data --threshold 0.15 --model-path /models/new.pth
```

Use non-default config files:

```bash
python main.py --mode live --config /path/to/config.yaml --runtime-config /path/to/runtime_config.yaml
```

## AWS IoT payload

When `--send-to-aws` is used, the app enables publishing at runtime and sends a payload like this:

```json
{
  "deviceId": "raspi-lab01",
  "timestamp": 1710000000,
  "result": "NORMAL",
  "error": 0.03421,
  "modelRun": "model_best_v1"
}
```

Fields come from [main.py](main.py) and [runtime_config.yaml](runtime_config.yaml). `timestamp` is a Unix timestamp in UTC seconds.

## Hardware assumptions

This project is set up for a Raspberry Pi deployment with:

- a USB microphone or USB audio interface
- an optional HD44780 16x2 LCD with PCF8574 I2C backpack
- optional AWS IoT Core connectivity

You will likely need to adjust these values in [runtime_config.yaml](runtime_config.yaml):

- `recording.card_index`
- `recording.mic_volume`
- `inference.model_path`
- `lcd.i2c_addr` and `lcd.i2c_bus`
- all certificate and endpoint values in `aws_iot`

## Output behavior

- Valid segments are shown on the LCD as `Valid audio` before inference
- Predictions are printed to the terminal with the reconstruction error
- In live mode, invalid segments are discarded and not passed to the model
- In batch mode, the script prints a final summary with totals

## Project structure

```text
project/
|-- main.py
|-- config.yaml
|-- runtime_config.yaml
|-- config_loader.py
|-- raspi_publish.py
|-- audio/
|   |-- recorder.py
|   |-- preprocess.py
|   `-- features.py
|-- inference/
|   `-- detector.py
|-- display/
|   `-- lcd.py
`-- models/
    |-- conv_autoencoder.py
    `-- model_best.pth
```

## Notes

- This repository contains the inference-side pipeline and model artifact. Training parameters still exist in [config.yaml](config.yaml), but training scripts are not included here.
- The current detector loads the convolutional autoencoder defined in [models/conv_autoencoder.py](models/conv_autoencoder.py).
