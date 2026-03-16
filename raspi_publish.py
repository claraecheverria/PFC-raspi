"""Helpers for publishing inference results to AWS IoT Core."""

from __future__ import annotations

import json
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

        self.connect()
        self._client.publish(self.topic, json.dumps(payload), self.qos)
        print(f"Published to AWS IoT: {payload}")
        return True

    def disconnect(self) -> None:
        if self._client is None:
            return

        self._client.disconnect()
        self._client = None
