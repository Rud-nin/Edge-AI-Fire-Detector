# ---------------------------------------------------------------------------
# Live test against the ESP32 firmware's HTTP image server.
#
# The board must be running a build with ENABLE_HTTP_SERVER (firmware/main/
# firmware.cpp) and be reachable on the network; its boot log prints the
# address ("Got IP: ..."). This script polls GET /image, decodes each frame
# (RGB565 or JPEG, per the X-Image-Format header) and runs the model on it, so
# it exercises exactly what the device captures - colour order included.
#
# When the model's class 1 wins it logs a line and paints a small red square in
# the top-left of the preview. Press 'q' in the window to quit. The poll rate is
# capped by HTTP_TARGET_FPS (test_common.py).
#
# Usage:
#     python test_espcam.py --input <model.pt> --address <ip> [options]
#
# Options:
#     --input PATH        required: trained checkpoint (.pt); must exist
#     --address STR       required: board IPv4 address (e.g. 192.168.1.42)
#     --port INT          HTTP port of the image server (default: 80)
#     --window-name STR   preview window title (default: model test (esp-cam))
#
# Examples:
#     python test_espcam.py --input ../models/Kairo_Quell.pt --address 192.168.1.42
#     python test_espcam.py --input ../models/Kairo_Quell.pt --address 10.0.0.5 --port 8080
# ---------------------------------------------------------------------------

import argparse
import ipaddress

import torch.nn as nn

from const import configure_logging, log
from test_common import HttpImageSource, load_model, run_inference
from utils import existing_file

PORT: int = 80
WINDOW_NAME: str = "model test (esp-cam)"


def ipv4_address(value: str) -> str:
    """argparse type: a valid IPv4 address."""
    try:
        return str(ipaddress.IPv4Address(value))
    except ipaddress.AddressValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid IPv4 address: {value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained model on the ESP32 camera stream.")
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
        help="trained checkpoint (.pt) to run",
    )
    parser.add_argument(
        "--address",
        type=ipv4_address,
        required=True,
        help="board IPv4 address (e.g. 192.168.1.42)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"HTTP port of the image server (default: {PORT})",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default=WINDOW_NAME,
        help=f"preview window title (default: {WINDOW_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging()

    url: str = f"http://{args.address}:{args.port}/image"
    log.info("input=%s url=%s window=%s", args.input, url, args.window_name)

    model: nn.Module = load_model(args.input)

    source: HttpImageSource = HttpImageSource(url)
    run_inference(model, source, args.window_name)


if __name__ == "__main__":
    main()
