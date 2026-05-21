"""
gpio_switch_handler.py - Jetson.GPIO button handler for the GELLO controller.

The pins are configured in Jetson.GPIO BOARD mode, so pin numbers refer to the
physical 40-pin header numbers. The three GELLO buttons are expected on
physical pins 7, 11, and 13.

Hardware assumption
-------------------
- Each button line is externally pulled HIGH.
- Pressing the button connects the line to GND.
- Released: HIGH
- Pressed : LOW
- Event   : falling edge, published as Bool(data=true)
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List


class GpioSwitchHandler:
    """
    Watches BOARD pins and emits one true pulse per button press.

    Parameters
    ----------
    pin_configs:
        List of (board_pin, topic_name), for example
        [(7, "/gello/switch/record"), ...].
    callback:
        Called as callback(topic_name, True) when a falling edge is accepted.
    pin_mode:
        Jetson.GPIO numbering mode. Use "BOARD" for physical header pins.
    bouncetime_ms:
        Per-pin cooldown window in milliseconds.
    confirm_delay_ms:
        Delay after the edge before confirming that the pin is still LOW.
    logger:
        Optional ROS logger.
    """

    def __init__(
        self,
        pin_configs: List[tuple],
        callback: Callable[[str, bool], None],
        pin_mode: str = "BOARD",
        bouncetime_ms: int = 250,
        confirm_delay_ms: int = 20,
        logger=None,
    ) -> None:
        self._pin_to_topic: Dict[int, str] = {
            int(pin): topic for pin, topic in pin_configs
        }
        self._callback = callback
        self._pin_mode = pin_mode
        self._bouncetime_s = max(0, bouncetime_ms) / 1000.0
        self._confirm_delay_s = max(0, confirm_delay_ms) / 1000.0
        self._logger = logger
        self._last_press_time: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._active = False

        try:
            import Jetson.GPIO as GPIO
        except ImportError:
            self._GPIO = None
            self._log_warn(
                "Jetson.GPIO not found. Install with: pip install Jetson.GPIO"
            )
            return

        self._GPIO = GPIO
        self._setup()

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #

    def _setup(self) -> None:
        GPIO = self._GPIO
        try:
            mode = GPIO.BOARD if self._pin_mode.upper() == "BOARD" else GPIO.BCM
            GPIO.setmode(mode)
        except Exception as exc:
            self._log_warn(f"GPIO.setmode failed: {exc}")
            return

        for pin, topic_name in self._pin_to_topic.items():
            try:
                GPIO.setup(pin, GPIO.IN)
                GPIO.add_event_detect(
                    pin,
                    GPIO.FALLING,
                    callback=self._edge_cb,
                    bouncetime=int(self._bouncetime_s * 1000),
                )
                self._log_info(
                    f"BOARD pin {pin} -> press event -> '{topic_name}'"
                )
            except Exception as exc:
                self._log_warn(
                    f"Failed to set up BOARD pin {pin} for topic "
                    f"'{topic_name}': {exc}"
                )

        self._active = True

    def cleanup(self) -> None:
        if self._GPIO is None or not self._active:
            return
        try:
            for pin in self._pin_to_topic:
                try:
                    self._GPIO.remove_event_detect(pin)
                except Exception:
                    pass
            self._GPIO.cleanup(list(self._pin_to_topic.keys()))
        except Exception as exc:
            self._log_warn(f"GPIO cleanup error: {exc}")
        self._active = False

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #

    def _edge_cb(self, channel: int) -> None:
        try:
            pin = int(channel)
            topic_name = self._pin_to_topic.get(pin)
            if topic_name is None:
                return

            now = time.monotonic()
            with self._lock:
                last = self._last_press_time.get(pin, 0.0)
                if now - last < self._bouncetime_s:
                    return
                self._last_press_time[pin] = now

            if self._confirm_delay_s > 0:
                time.sleep(self._confirm_delay_s)
            if self._GPIO.input(pin) != self._GPIO.LOW:
                return

            self._callback(topic_name, True)
        except Exception as exc:
            self._log_warn(f"GPIO edge callback error on channel {channel}: {exc}")

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #

    def _log_info(self, msg: str) -> None:
        if self._logger:
            self._logger.info(f"[GpioSwitchHandler] {msg}")
        else:
            print(f"[GpioSwitchHandler] {msg}")

    def _log_warn(self, msg: str) -> None:
        if self._logger:
            self._logger.warn(f"[GpioSwitchHandler] {msg}")
        else:
            print(f"[GpioSwitchHandler] WARNING: {msg}")
