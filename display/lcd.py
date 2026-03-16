"""display/lcd.py

HD44780 LCD over I2C (PCF8574 backpack) using direct smbus2 bit-banging.
Same low-level driver as the original lcd_display.py, wrapped in a class.

If smbus2 is not installed or the display is not connected every call
becomes a no-op so the rest of the system keeps running unaffected.
"""

from __future__ import annotations

import time

try:
    import smbus2
    _HAS_SMBUS = True
except ImportError:
    _HAS_SMBUS = False


# ── HD44780 constants ─────────────────────────────────────────────────────────
_LCD_CHR       = 1           # character mode
_LCD_CMD       = 0           # command mode
_LCD_LINE_1    = 0x80
_LCD_LINE_2    = 0xC0
_LCD_BACKLIGHT = 0x08
_ENABLE        = 0b00000100


class LCDDisplay:
    """High-level LCD wrapper backed by direct smbus2 I2C bit-banging.

    Args:
        cfg: Full config dict.  Uses the 'lcd' section.
    """

    _STATUS_MAP: dict[str, tuple[str, str]] = {
        "off machine": ("Machine OFF",     "Check system"),
        "loud noise":  ("LOUD NOISE!",      "External event"),
        "valid":       ("Valid audio",       "Recording OK"),
        "normal":      ("Result: NORMAL",    ""),
        "anomaly":     ("ANOMALY DETECTED",  "Check machine!"),
    }

    def __init__(self, cfg: dict):
        lcd_cfg      = cfg.get("lcd", {})
        self.enabled = bool(lcd_cfg.get("enabled", True)) and _HAS_SMBUS
        self.width   = int(lcd_cfg.get("width",    16))
        self._addr   = int(lcd_cfg.get("i2c_addr", 0x27))
        self._bus_id = int(lcd_cfg.get("i2c_bus",  1))
        self._bus    = None

        if self.enabled:
            self._init_bus()
            self._lcd_init()

    # ── public ───────────────────────────────────────────────────────────────

    def show(self, status: str) -> None:
        """Display a predefined status.

        Known keys: "off machine", "loud noise", "valid", "normal", "anomaly".
        Unknown keys show "Unknown state".
        """
        if not self.enabled:
            return
        line1, line2 = self._STATUS_MAP.get(status.lower(), ("Unknown state", ""))
        self._lcd_init()
        self._lcd_string(line1, _LCD_LINE_1)
        self._lcd_string(line2, _LCD_LINE_2)

    def show_error(self, error: float, label: str) -> None:
        """Display reconstruction error and prediction label."""
        if not self.enabled:
            return
        self._lcd_init()
        self._lcd_string(f"Err:{error:.5f}", _LCD_LINE_1)
        self._lcd_string(label, _LCD_LINE_2)

    def clear(self) -> None:
        if not self.enabled:
            return
        self._lcd_init()
        self._lcd_string("", _LCD_LINE_1)
        self._lcd_string("", _LCD_LINE_2)

    # ── low-level driver (mirrors original lcd_display.py exactly) ───────────

    def _init_bus(self) -> None:
        try:
            self._bus = smbus2.SMBus(self._bus_id)
        except Exception as exc:
            print(f"LCD bus init failed: {exc}  (display disabled)")
            self.enabled = False

    def _lcd_init(self) -> None:
        """Initialise the HD44780 controller (4-bit mode)."""
        if self._bus is None:
            return
        try:
            for cmd in (0x33, 0x32, 0x06, 0x0C, 0x28, 0x01):
                self._lcd_byte(cmd, _LCD_CMD)
            time.sleep(0.005)
        except Exception as exc:
            print(f"LCD init failed: {exc}  (display disabled)")
            self.enabled = False

    def _lcd_byte(self, bits: int, mode: int) -> None:
        """Send one byte to the LCD in two nibbles."""
        bits_high = mode | (bits & 0xF0)        | _LCD_BACKLIGHT
        bits_low  = mode | ((bits << 4) & 0xF0) | _LCD_BACKLIGHT
        self._bus.write_byte(self._addr, bits_high)
        self._toggle_enable(bits_high)
        self._bus.write_byte(self._addr, bits_low)
        self._toggle_enable(bits_low)

    def _toggle_enable(self, bits: int) -> None:
        time.sleep(0.0005)
        self._bus.write_byte(self._addr, bits | _ENABLE)
        time.sleep(0.0005)
        self._bus.write_byte(self._addr, bits & ~_ENABLE)
        time.sleep(0.0005)

    def _lcd_string(self, message: str, line: int) -> None:
        """Write a padded string to the given line address."""
        message = message.ljust(self.width, " ")[:self.width]
        self._lcd_byte(line, _LCD_CMD)
        for char in message:
            self._lcd_byte(ord(char), _LCD_CHR)
