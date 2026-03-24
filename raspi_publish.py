"""Helpers for publishing inference results to AWS IoT Core."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


class AWSIoTPublisher:
    """Lazy AWS IoT client used by the main inference pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.device_id = cfg.get("device_id", "raspi-lab01")
        self.model_run = cfg.get("model_run", "default")
        self.endpoint = cfg.get("endpoint")
        self.topic = cfg.get("topic")
        self.port = int(cfg.get("port", 8883))
        self.root_ca = cfg.get("root_ca")
        self.cert = cfg.get("cert")
        self.private_key = cfg.get("private_key")
        self.qos = int(cfg.get("qos", 1))
        self.offline_queue_size = int(cfg.get("offline_queue_size", -1))
        self.draining_frequency = int(cfg.get("draining_frequency", 2))
        self.connect_disconnect_timeout = int(
            cfg.get("connect_disconnect_timeout", 10)
        )
        self.operation_timeout = int(cfg.get("operation_timeout", 5))
        self.backup_enabled = bool(cfg.get("backup_enabled", True))
        default_backup_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "var",
            "aws_iot_pending.jsonl",
        )
        self.backup_path = os.path.abspath(
            os.path.expanduser(cfg.get("backup_path", default_backup_path))
        )
        self.backup_max_messages = int(cfg.get("backup_max_messages", 1000))
        self.backup_flush_batch_size = int(cfg.get("backup_flush_batch_size", 50))
        self._client = None

    def _validate(self) -> None:
        missing = [
            name
            for name, value in {
                "endpoint": self.endpoint,
                "topic": self.topic,
                "root_ca": self.root_ca,
                "cert": self.cert,
                "private_key": self.private_key,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing AWS IoT config fields: {', '.join(sorted(missing))}"
            )

    def connect(self) -> None:
        if not self.enabled or self._client is not None:
            return

        self._validate()

        try:
            from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
        except ImportError as exc:
            raise RuntimeError(
                "AWSIoTPythonSDK is not installed, cannot publish to AWS IoT."
            ) from exc

        client = AWSIoTMQTTClient(self.device_id)
        client.configureEndpoint(self.endpoint, self.port)
        client.configureCredentials(self.root_ca, self.private_key, self.cert)
        client.configureOfflinePublishQueueing(self.offline_queue_size)
        client.configureDrainingFrequency(self.draining_frequency)
        client.configureConnectDisconnectTimeout(self.connect_disconnect_timeout)
        client.configureMQTTOperationTimeout(self.operation_timeout)

        print("Connecting to AWS IoT...")
        client.connect()
        print("Connected to AWS IoT.")
        self._client = client

    def publish(self, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        self.flush_pending()

        try:
            self.connect()
            self._publish_now(payload)
            return True
        except Exception as exc:
            self._reset_client()
            self._backup_payload(payload)
            raise RuntimeError(
                f"{exc} | payload stored locally at {self.backup_path}"
            ) from exc

    def flush_pending(self) -> int:
        if not self.enabled or not self.backup_enabled:
            return 0

        pending = self._load_pending()
        if not pending:
            return 0

        try:
            self.connect()
        except Exception as exc:
            print(f"AWS IoT still offline, keeping {len(pending)} queued payload(s): {exc}")
            self._reset_client()
            return 0

        sent = 0
        for index, payload in enumerate(pending[: self.backup_flush_batch_size]):
            try:
                self._publish_now(payload)
                sent += 1
            except Exception as exc:
                self._reset_client()
                self._write_pending(pending[index:])
                print(
                    "AWS IoT reconnect failed while draining local backup, "
                    f"keeping {len(pending) - index} queued payload(s): {exc}"
                )
                return sent

        remaining = pending[sent:]
        self._write_pending(remaining)
        if sent:
            print(
                f"Flushed {sent} queued AWS IoT payload(s); "
                f"{len(remaining)} remaining on disk."
            )
        return sent

    def disconnect(self) -> None:
        if self._client is None:
            return

        self._reset_client()

    def _publish_now(self, payload: Dict[str, Any]) -> None:
        self._client.publish(self.topic, json.dumps(payload), self.qos)
        print(f"Published to AWS IoT: {payload}")

    def _backup_payload(self, payload: Dict[str, Any]) -> None:
        if not self.backup_enabled:
            return

        pending = self._load_pending()
        pending.append(payload)

        if self.backup_max_messages > 0 and len(pending) > self.backup_max_messages:
            dropped = len(pending) - self.backup_max_messages
            pending = pending[-self.backup_max_messages :]
            print(
                "Local AWS backup is full, dropping "
                f"{dropped} oldest payload(s)."
            )

        self._write_pending(pending)
        print(f"Saved payload to local AWS backup: {self.backup_path}")

    def _load_pending(self) -> list[Dict[str, Any]]:
        if not self.backup_enabled or not os.path.exists(self.backup_path):
            return []

        pending: list[Dict[str, Any]] = []
        with open(self.backup_path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    pending.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        "Skipping malformed AWS backup entry at "
                        f"{self.backup_path}:{line_number}"
                    )
        return pending

    def _write_pending(self, payloads: list[Dict[str, Any]]) -> None:
        backup_dir = os.path.dirname(self.backup_path)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)

        if not payloads:
            if os.path.exists(self.backup_path):
                os.remove(self.backup_path)
            return

        temp_path = f"{self.backup_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload))
                fh.write("\n")
        os.replace(temp_path, self.backup_path)

    def _reset_client(self) -> None:
        if self._client is None:
            return

        try:
            self._client.disconnect()
        except Exception:
            pass
        finally:
            self._client = None
