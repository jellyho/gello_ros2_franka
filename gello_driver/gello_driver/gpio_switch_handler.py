"""
gpio_switch_handler.py — Jetson GPIO switch input handler.

Monitors configurable GPIO pins (hardware pull-up assumed) for rising
edges and publishes a std_msgs/Bool (True) on individual ROS 2 topics.

Hardware assumption
-------------------
- Each pin is pulled HIGH by an external resistor.
- Switch connects the pin to GND.
- Rest state  : pin = HIGH
- Pressed     : pin = LOW
- Rising edge : LOW → HIGH  (button release)

If you want falling edge (button press detection) instead, change
GPIO.RISING to GPIO.FALLING in _setup_pin() and update the comment.

Jetson.GPIO notes
-----------------
- Library: `Jetson.GPIO` (pip install Jetson.GPIO)
- Default numbering: BOARD (physical pin number, stable across boots)
- Run the node with sufficient privileges or add the user to the
  `gpio` group:  sudo usermod -aG gpio $USER
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional


class GpioSwitchHandler:
    """
    Manages rising-edge detection on a set of GPIO pins.

    Parameters
    ----------
    pin_configs : list of (pin_number, topic_name)
        Each entry maps one physical (BOARD) pin to the ROS 2 topic name
        that should be published when a rising edge is detected.
    callback : Callable[[str], None]
        Called with the topic_name whenever a rising edge fires on the
        associated pin.
    pin_mode : str
        'BOARD' (physical pin numbers, recommended) or 'BCM'.
    bouncetime_ms : int
        Debounce time in milliseconds (default 50 ms).
    logger : optional ROS 2 logger
    """

    def __init__(
        self,
        pin_configs: List[tuple],   # [(pin_number, topic_name), ...]
        callback: Callable[[str], None],
        pin_mode: str = "BOARD",
        bouncetime_ms: int = 50,
        logger=None,
    ) -> None:
        self._pin_configs = pin_configs          # [(pin, topic_name)]
        self._callback = callback
        self._bouncetime = bouncetime_ms
        self._logger = logger
        self._active = False
        self._lock = threading.Lock()

        # Import Jetson.GPIO
        try:
            import Jetson.GPIO as GPIO
            self._GPIO = GPIO
        except ImportError:
            if logger:
                logger.warn(
                    "[GpioSwitchHandler] Jetson.GPIO not found. "
                    "GPIO switches will be DISABLED. "
                    "Install with: pip install Jetson.GPIO"
                )
            self._GPIO = None
            return

        self._setup(pin_mode)

    # ----------------------------------------------------------------------- #
    # Setup / teardown
    # ----------------------------------------------------------------------- #

    def _setup(self, pin_mode: str) -> None:
        GPIO = self._GPIO

        try:
            mode = GPIO.BOARD if pin_mode.upper() == "BOARD" else GPIO.BCM
            GPIO.setmode(mode)
        except Exception as exc:
            self._log_warn(f"GPIO.setmode failed: {exc}")
            self._GPIO = None
            return

        for pin, topic in self._pin_configs:
            try:
                # INPUT only — hardware pull-up is external, so PUD_OFF
                GPIO.setup(pin, GPIO.IN)
                GPIO.add_event_detect(
                    pin,
                    GPIO.RISING,
                    callback=self._make_edge_callback(topic),
                    bouncetime=self._bouncetime,
                )
                self._log_info(
                    f"GPIO pin {pin} → rising edge → '{topic}'"
                )
            except Exception as exc:
                self._log_warn(
                    f"Failed to set up GPIO pin {pin} for topic '{topic}': {exc}"
                )

        self._active = True

    def cleanup(self) -> None:
        """Release all GPIO resources. Call on node shutdown."""
        if self._GPIO is None or not self._active:
            return
        try:
            for pin, _ in self._pin_configs:
                try:
                    self._GPIO.remove_event_detect(pin)
                except Exception:
                    pass
            self._GPIO.cleanup([pin for pin, _ in self._pin_configs])
        except Exception as exc:
            self._log_warn(f"GPIO cleanup error: {exc}")
        self._active = False

    # ----------------------------------------------------------------------- #
    # Edge callback factory
    # ----------------------------------------------------------------------- #

    def _make_edge_callback(self, topic_name: str) -> Callable:
        """Return a GPIO interrupt callback bound to the given topic_name."""

        def _cb(channel: int) -> None:
            # GPIO callbacks fire in a C thread; avoid heavy work here
            try:
                self._callback(topic_name)
            except Exception as exc:
                self._log_warn(f"Edge callback error on pin {channel}: {exc}")

        return _cb

    # ----------------------------------------------------------------------- #
    # Logging helpers
    # ----------------------------------------------------------------------- #

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
