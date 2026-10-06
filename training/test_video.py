# ---------------------------------------------------------------------------
# Offline video test for a trained (un-quantized) checkpoint.
#
# Plays a video file, shows it in a window and runs the model on every frame.
# When the model's class 1 wins it logs a line and paints a small red square in
# the top-left of the preview. Press 'q' to quit early; playback otherwise
# stops at the end of the video.
#
# Option A preprocessing is used (raw RGB in [0, 255]), the same as training, so
# the result should match the model. Large videos are downscaled for display to
# DISPLAY_MAX_SIZE (const.py), so the window never gets huge.
#
# Usage:
#     python test_video.py --input <model.pt> --video <video file> [options]
#
# Options:
#     --input PATH        required: trained checkpoint (.pt); must exist
#     --video PATH        required: video file to play; must exist
#     --window-name STR   preview window title (default: model test (video))
#
# Examples:
#     python test_video.py --input ../models/Kairo_Quell.pt --video ~/clips/fire.mp4
#     python test_video.py --input ../models/Kairo_Quell.pt --video clip.mp4 \
#         --window-name fire
# ---------------------------------------------------------------------------

import argparse
import sys

import cv2
import torch.nn as nn

from const import configure_logging, log
from test_common import CaptureSource, load_model, run_inference
from utils import existing_file

WINDOW_NAME: str = "model test (video)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained model on a video file.")
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
        help="trained checkpoint (.pt) to run",
    )
    parser.add_argument(
        "--video",
        type=existing_file,
        required=True,
        help="video file to play",
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

    log.info("input=%s video=%s window=%s", args.input, args.video, args.window_name)

    model: nn.Module = load_model(args.input)

    capture: cv2.VideoCapture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        log.error("could not open video=%s", args.video)
        sys.exit(1)
    log.info("opened video=%s", args.video)

    run_inference(model, CaptureSource(capture), args.window_name, end_message="end of video")


if __name__ == "__main__":
    main()
