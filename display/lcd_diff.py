"""display/lcd.py

HD44780 LCD over I²C (PCF8574 backpack).

If the required libraries are not installed or the display is not
connected, all calls become no-ops so the rest of the system keeps
running unaffected.
"""

from __future__ import annotations

import time

try:
    import smbus2
    from RPLCD.i2c import CharLCD as _CharLCD
    _HAS_LCD = True
except ImportError:
    _HAS_LCD = False


class LCDDisplay:
    """High-level LCD wrapper.

    Args:
        cfg: Full config dict.  Uses the 'lcd' section.
    """

    # Status → (line1, line2)
    _STATUS_MAP: dict[str, tuple[str, str]] = {
        "off machine": ("Machine OFF",      "Check system"),
        "loud noise":  ("LOUD NOISE!",       "External event"),
        "valid":       ("Valid audio",        "Recording OK"),
        "normal":      ("Result: NORMAL",     ""),
        "anomaly":     ("ANOMALY DETECTED",   "Check machine!"),
    }

    def __init__(self, cfg: dict):
        lcd_cfg      = cfg.get("lcd", {})
        self.enabled = bool(lcd_cfg.get("enabled", True)) and _HAS_LCD
        self.width   = int(lcd_cfg.get("width", 16))
        self._addr   = int(lcd_cfg.get("i2c_addr", 0x27))
        self._bus    = int(lcd_cfg.get("i2c_bus",  1))
        self._lcd    = None

        if self.enabled:
            self._init_lcd()

    # ── public ───────────────────────────────────────────────

    def show(self, status: str) -> None:
        """Display a predefined status message.

        Known keys: "off machine", "loud noise", "valid", "normal", "anomaly".
        Unknown keys show "Unknown state".
        """
        if not self.enabled:
            return
        line1, line2 = self._STATUS_MAP.get(status.lower(), ("Unknown state", ""))
        self._write(line1, line2)

    def show_error(self, error: float, label: str) -> None:
        """Display the reconstruction error and prediction label."""
        if not self.enabled:
            return
        self._write(
            f"Err:{error:.5f}",
            label[:self.width],
        )

    def clear(self) -> None:
        if not self.enabled:
            return
        self._write("", "")

    # ── private ──────────────────────────────────────────────

    def _init_lcd(self) -> None:
        try:
            self._lcd = _CharLCD(
                i2c_expander="PCF8574",
                address=self._addr,
                port=self._bus,
                cols=self.width,
                rows=2,
                dotsize=8,
            )
            self._lcd.clear()
        except Exception as exc:
            print(f"⚠  LCD init failed: {exc}  (display disabled)")
            self.enabled = False

    def _write(self, line1: str, line2: str) -> None:
        if self._lcd is None:
            return
        try:
            self._lcd.clear()
            self._lcd.write_string(line1.ljust(self.width)[:self.width])
            self._lcd.crlf()
            self._lcd.write_string(line2.ljust(self.width)[:self.width])
        except Exception as exc:
            print(f"⚠  LCD write error: {exc}")
