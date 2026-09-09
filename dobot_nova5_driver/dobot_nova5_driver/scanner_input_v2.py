"""Linux input reader copied from hidpospython, without alarm-light support."""

from __future__ import annotations

import fcntl
import glob
import logging
import os
import struct
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

EV_KEY = 0x01
KEY_PRESSED = 1
KEY_REPEATED = 2
KEY_ENTER = 28
KEY_KPENTER = 96
KEY_LEFTSHIFT = 42
KEY_RIGHTSHIFT = 54
INPUT_EVENT_FORMAT = "llHHI"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FORMAT)
EVIOCGRAB = 0x40044590

KEY_MAP = {
    2: ("1", "!"), 3: ("2", "@"), 4: ("3", "#"), 5: ("4", "$"),
    6: ("5", "%"), 7: ("6", "^"), 8: ("7", "&"), 9: ("8", "*"),
    10: ("9", "("), 11: ("0", ")"), 12: ("-", "_"), 13: ("=", "+"),
    16: ("q", "Q"), 17: ("w", "W"), 18: ("e", "E"), 19: ("r", "R"),
    20: ("t", "T"), 21: ("y", "Y"), 22: ("u", "U"), 23: ("i", "I"),
    24: ("o", "O"), 25: ("p", "P"), 26: ("[", "{"), 27: ("]", "}"),
    30: ("a", "A"), 31: ("s", "S"), 32: ("d", "D"), 33: ("f", "F"),
    34: ("g", "G"), 35: ("h", "H"), 36: ("j", "J"), 37: ("k", "K"),
    38: ("l", "L"), 39: (";", ":"), 40: ("'", '"'), 41: ("`", "~"),
    43: ("\\", "|"), 44: ("z", "Z"), 45: ("x", "X"), 46: ("c", "C"),
    47: ("v", "V"), 48: ("b", "B"), 49: ("n", "N"), 50: ("m", "M"),
    51: (",", "<"), 52: (".", ">"), 53: ("/", "?"), 55: ("*", "*"),
    57: (" ", " "), 71: ("7", "7"), 72: ("8", "8"), 73: ("9", "9"),
    74: ("-", "-"), 75: ("4", "4"), 76: ("5", "5"), 77: ("6", "6"),
    78: ("+", "+"), 79: ("1", "1"), 80: ("2", "2"), 81: ("3", "3"),
    82: ("0", "0"), 83: (".", "."), 98: ("/", "/"),
}

PREFERRED_DEVICE_KEYWORDS = ("BTW_Hid_Device", "Hid_Device", "Barcode", "Scanner")
EXCLUDED_DEVICE_KEYWORDS = ("ITE_Device", "USB_Receiver")


def find_scanner_event_device() -> Optional[str]:
    candidates = sorted(glob.glob("/dev/input/by-id/*event-kbd"))
    for path in candidates:
        basename = os.path.basename(path)
        if any(word in basename for word in EXCLUDED_DEVICE_KEYWORDS):
            continue
        if any(word in basename for word in PREFERRED_DEVICE_KEYWORDS):
            return path
    return None


class ScannerInputReader:
    def __init__(self, device_path: str, callback: Callable[[str], None], grab_device: bool = True):
        self.device_path = device_path
        self.callback = callback
        self.grab_device = grab_device
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.buffer = ""
        self.shift_pressed = False
        self.last_key_time = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _read_loop(self) -> None:
        try:
            with open(self.device_path, "rb") as event_file:
                if self.grab_device:
                    try:
                        fcntl.ioctl(event_file.fileno(), EVIOCGRAB, 1)
                        logger.info("Scanner keyboard output grabbed exclusively")
                    except OSError as exc:
                        logger.warning("Could not grab scanner input exclusively: %s", exc)
                logger.info("Reading scanner input from %s", self.device_path)
                while self.running:
                    event_data = event_file.read(INPUT_EVENT_SIZE)
                    if len(event_data) != INPUT_EVENT_SIZE:
                        time.sleep(0.01)
                        continue
                    _, _, event_type, event_code, event_value = struct.unpack(INPUT_EVENT_FORMAT, event_data)
                    self._handle_event(event_type, event_code, event_value)
        except PermissionError as exc:
            logger.error("No permission to read scanner %s: %s", self.device_path, exc)
        except OSError as exc:
            if self.running:
                logger.error("Scanner read failed for %s: %s", self.device_path, exc)
        finally:
            try:
                if self.grab_device and "event_file" in locals() and not event_file.closed:
                    fcntl.ioctl(event_file.fileno(), EVIOCGRAB, 0)
            except OSError:
                pass
            self.running = False

    def _handle_event(self, event_type: int, event_code: int, event_value: int) -> None:
        if event_type != EV_KEY or event_value == KEY_REPEATED:
            return
        if event_code in (KEY_LEFTSHIFT, KEY_RIGHTSHIFT):
            self.shift_pressed = event_value == KEY_PRESSED
            return
        if event_value != KEY_PRESSED:
            return
        now = time.monotonic()
        if now - self.last_key_time > 0.3:
            self.buffer = ""
        self.last_key_time = now
        if event_code in (KEY_ENTER, KEY_KPENTER):
            value = self.buffer.strip()
            self.buffer = ""
            if len(value) >= 4:
                self.callback(value)
            return
        key_value = KEY_MAP.get(event_code)
        if key_value:
            self.buffer += key_value[1] if self.shift_pressed else key_value[0]

