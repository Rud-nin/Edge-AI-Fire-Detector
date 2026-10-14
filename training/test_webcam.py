# ---------------------------------------------------------------------------
# Live webcam test for a trained (un-quantized) checkpoint.
#
# Opens the local camera, shows it in a window and runs the model on every
# frame. When the model's class 1 wins it logs a line and paints a small red
# square in the top-left of the preview. Press 'q' in the window to quit.
#
# Input preparation follows the model's DATA_FLAGS (const.py): raw RGB in
# [0, 255] for Model1..Model4, the ImageNet mean/std shift for Model5 - the same
# as training, so the result should match the model. For the quantized/on-device
# version use test_espcam.py instead.
#
# Usage:
#     python test_webcam.py --model-type N --input <model.pt> [options]
#
# Options:
#     --model-type INT    required: build ModelN from model.py
#     --input PATH        required: trained checkpoint (.pt); must exist
#     --camera-index INT  camera to open (default: 0)
#     --window-name STR   preview window title (default: model test (webcam))
#
# Examples:
#     python test_webcam.py --model-type 2 --input ../models/Kairo_Quell.pt
#     python test_webcam.py --model-type 2 --input ../models/Kairo_Quell.pt --camera-index 1
# ---------------------------------------------------------------------------

import argparse
import sys

import cv2
import torch.nn as nn

from const import configure_logging, log
from model import MODEL_CLASSES
from test_common import CaptureSource, load_model, run_inference
from utils import existing_file

CAMERA_INDEX: int = 0
WINDOW_NAME: str = "model test (webcam)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained model on the local webcam.")
    parser.add_argument(
        "--model-type",
        type=int,
        required=True,
        choices=sorted(MODEL_CLASSES),
        help="model to run: N builds ModelN from model.py (e.g. 5 -> Model5)",
    )
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
        help="trained checkpoint (.pt) to run",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help=f"camera index to open (default: {CAMERA_INDEX})",
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

    log.info(
        "model_type=%d input=%s camera_index=%d window=%s",
        args.model_type, args.input, args.camera_index, args.window_name,
    )

    model: nn.Module = load_model(args.input, args.model_type)

    capture: cv2.VideoCapture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        log.error("could not open camera index=%d", args.camera_index)
        sys.exit(1)

    run_inference(model, CaptureSource(capture), args.window_name, end_message="camera read failed")


if __name__ == "__main__":
    main()
