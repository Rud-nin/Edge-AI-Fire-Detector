#!/usr/bin/env python3
"""Real-time viewer for the firmware's WiFi image server.

The firmware exposes a single endpoint:

    GET /image
      body    raw image bytes (RGB565, width*height*2, little-endian)
      headers X-Image-Width, X-Image-Height, X-Image-Format

This script polls that endpoint on a loop and displays the frames in an OpenCV
window.

Install:
    pip install -r tools/requirements.txt

Usage:
    python tools/monitor.py 192.168.1.42
    python tools/monitor.py 192.168.1.42 -p 8080
    python tools/monitor.py 192.168.1.42 --stats 2 --scale 1.0 --swap-rb

Every --stats seconds it prints received bytes and throughput to stderr, e.g.:
    rx=1,382,400 B  speed=153,600 B/s (150.0 KiB/s)  fps=1.0  frames=9 errors=0
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import numpy as np
except ImportError:
    _missing = ["numpy"]
else:
    _missing = []

try:
    import requests
except ImportError:
    _missing.append("requests")

try:
    import cv2
except ImportError:
    _missing.append("opencv-python")

if _missing:
    sys.exit(
        "missing Python package(s): " + ", ".join(_missing) + "\n"
        "  install them into the interpreter you run this with:\n"
        f"    {sys.executable} -m pip install -r tools/requirements.txt\n"
        "  (do NOT use sudo: it selects a different Python/site-packages)"
    )

IMAGE_PATH = "/image"


class Stats:
    """Counters plus interval throughput, for the periodic report."""

    def __init__(self):
        self.total_bytes = 0
        self.frames = 0
        self.errors = 0
        self._start = time.time()
        self._last_bytes = 0
        self._last_time = self._start

    def report(self):
        now = time.time()
        dt = max(now - self._last_time, 1e-9)
        speed = (self.total_bytes - self._last_bytes) / dt
        self._last_bytes = self.total_bytes
        self._last_time = now
        fps = self.frames / max(now - self._start, 1e-9)
        return (f"rx={self.total_bytes:,} B  speed={speed:,.0f} B/s "
                f"({speed / 1024:,.1f} KiB/s)  fps={fps:.1f}  "
                f"frames={self.frames} errors={self.errors}")


def decode(width, height, pixfmt, payload):
    """Decode a payload into an OpenCV BGR uint8 image."""
    fmt = (pixfmt or "rgb565").lower()
    if fmt == "rgb565":
        # Standard RGB565 read as little-endian uint16 (esp-dl RGB565LE).
        x = np.frombuffer(payload, dtype=">u2").reshape(height, width)
        r = ((x >> 11) & 0x1F).astype(np.uint16) << 3
        g = ((x >> 5) & 0x3F).astype(np.uint16) << 2
        b = (x & 0x1F).astype(np.uint16) << 3
        return np.dstack([b, g, r]).astype(np.uint8)

    rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    return rgb[:, :, ::-1].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Display the ESP32-S3 WiFi image stream in real time."
    )
    parser.add_argument("host", help="board IPv4 address (e.g. 192.168.1.42)")
    parser.add_argument("-p", "--port", type=int, default=80,
                        help="HTTP port (default: 80)")
    parser.add_argument("-s", "--scale", type=float, default=2.0,
                        help="window scale factor (default: 2.0)")
    parser.add_argument("--swap-rb", action="store_true",
                        help="swap red/blue if colours look wrong")
    parser.add_argument("--stats", type=float, default=1.0, metavar="SECONDS",
                        help="periodic rx/speed report interval (0 disables)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="HTTP timeout in seconds (default: 5.0)")
    parser.add_argument("--window", default="ESP32 image stream",
                        help="window title")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}{IMAGE_PATH}"
    print(f"polling {url} ... (press 'q' in the window to quit)", file=sys.stderr)

    session = requests.Session()
    stats = Stats()
    last_report = time.time()

    try:
        while True:
            try:
                resp = session.get(url, timeout=args.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                stats.errors += 1
                print(f"request failed: {exc}", file=sys.stderr)
                time.sleep(0.5)
                continue

            width = int(resp.headers.get("X-Image-Width", 0))
            height = int(resp.headers.get("X-Image-Height", 0))
            fmt = (resp.headers.get("X-Image-Format", "rgb565") or "rgb565").lower()
            payload = resp.content

            bpp = {"rgb565": 2, "rgb888": 3}.get(fmt)
            expected = width * height * bpp if bpp else 0
            if width <= 0 or height <= 0 or bpp is None or len(payload) != expected:
                stats.errors += 1
                print(f"bad frame: {width}x{height} {fmt}, {len(payload)} B "
                      f"(expected {expected} B)", file=sys.stderr)
                time.sleep(0.5)
                continue

            stats.total_bytes += len(payload)
            stats.frames += 1

            img = decode(width, height, fmt, payload)
            if args.swap_rb:
                img = img[:, :, ::-1].copy()
            if args.scale != 1.0:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)

            cv2.imshow(args.window, img)

            if args.stats > 0 and time.time() - last_report >= args.stats:
                print(stats.report(), file=sys.stderr)
                last_report = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    except cv2.error as exc:
        print(f"OpenCV display error: {exc}\n"
              f"  this needs a graphical display (not a headless SSH session).",
              file=sys.stderr)
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
