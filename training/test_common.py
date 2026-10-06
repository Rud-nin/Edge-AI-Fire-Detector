import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import Model4
from const import (
    CLASS_NAMES,
    DEVICE,
    DISPLAY_MAX_SIZE,
    INDICATOR_COLOR,
    INDICATOR_SIZE,
    eval_transform,
    log,
)

HTTP_TIMEOUT: float = 30.0
HTTP_RETRIES: int = 3
HTTP_RETRY_DELAY: float = 0.5
HTTP_TARGET_FPS: float = 10.0  # cap the poll rate (0 = unlimited)


class FrameSource(Protocol):
    """A source of BGR frames, one at a time (see CaptureSource/HttpImageSource)."""

    def read(self) -> Optional[np.ndarray]: ...
    def release(self) -> None: ...


class CaptureSource:
    """Frames from a cv2.VideoCapture (webcam or video file)."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture: cv2.VideoCapture = capture

    def read(self) -> Optional[np.ndarray]:
        ok: bool
        frame: np.ndarray
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        self.capture.release()


class HttpImageSource:
    """Frames pulled from the firmware's HTTP GET /image endpoint."""

    def __init__(
        self,
        url: str,
        timeout: float = HTTP_TIMEOUT,
        target_fps: float = HTTP_TARGET_FPS,
    ) -> None:
        self.url: str = url
        self.timeout: float = timeout
        self.min_interval: float = (1.0 / target_fps) if target_fps > 0.0 else 0.0
        self.session: requests.Session = requests.Session()
        self._last_read: float = 0.0

    def read(self) -> Optional[np.ndarray]:
        # Throttle so the client does not hammer the board (which also runs the
        # camera and, in model mode, the inference loop).
        if self.min_interval > 0.0:
            remaining: float = self.min_interval - (time.time() - self._last_read)
            if remaining > 0.0:
                time.sleep(remaining)
        self._last_read = time.time()

        # Retry a few times so a single hiccup does not end the preview.
        for attempt in range(HTTP_RETRIES):
            try:
                response: requests.Response = self.session.get(self.url, timeout=self.timeout)
                response.raise_for_status()
                return decode_image(
                    int(response.headers.get("X-Image-Width", 0)),
                    int(response.headers.get("X-Image-Height", 0)),
                    response.headers.get("X-Image-Format", "rgb565"),
                    response.content,
                )
            except (requests.RequestException, ValueError) as exc:
                log.warning("request %d/%d to %s failed: %s", attempt + 1, HTTP_RETRIES, self.url, exc)
                time.sleep(HTTP_RETRY_DELAY)
        return None

    def release(self) -> None:
        self.session.close()


def decode_image(width: int, height: int, pixfmt: str, payload: bytes) -> np.ndarray:
    """Decode an /image payload (rgb565, rgb888 or jpeg) into an OpenCV BGR frame."""
    fmt: str = (pixfmt or "rgb565").lower()

    if fmt == "rgb565":
        # The board's RGB565 is big-endian uint16 (matches esp-dl RGB565BE).
        x: np.ndarray = np.frombuffer(payload, dtype=">u2").reshape(height, width)
        r: np.ndarray = ((x >> 11) & 0x1F).astype(np.uint16) << 3
        g: np.ndarray = ((x >> 5) & 0x3F).astype(np.uint16) << 2
        b: np.ndarray = (x & 0x1F).astype(np.uint16) << 3
        return np.dstack([b, g, r]).astype(np.uint8)

    if fmt == "jpeg":
        frame: Optional[np.ndarray] = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise ValueError("JPEG decode failed")
        return frame

    if fmt == "rgb888":
        rgb: np.ndarray = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        return rgb[:, :, ::-1].copy()

    raise ValueError(f"unsupported image format: {pixfmt}")


def preprocess(frame: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    # OpenCV delivers BGR; the model was trained on RGB PIL images.
    rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)


def fit_for_display(frame: np.ndarray) -> np.ndarray:
    # Downscale so the longest edge is at most DISPLAY_MAX_SIZE (never upscale).
    height, width = frame.shape[:2]
    scale: float = DISPLAY_MAX_SIZE / max(height, width)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def load_model(model_path: Path) -> nn.Module:
    model: nn.Module = Model4().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    log.info("loaded model=%s device=%s", model_path, DEVICE)
    return model


def run_inference(
    model: nn.Module,
    source: FrameSource,
    window_name: str,
    end_message: str = "stream ended",
) -> None:
    # read frame -> preprocess -> forward -> log + paint INDICATOR_* -> imshow.
    # Press 'q' in the window to stop early.
    try:
        with torch.no_grad():
            while True:
                frame: Optional[np.ndarray] = source.read()
                if frame is None:
                    log.info("%s", end_message)
                    break

                inputs: torch.Tensor = preprocess(frame, eval_transform)
                outputs: torch.Tensor = model(inputs)[0]

                # scale huge sources down so the window stays a sane size
                display: np.ndarray = fit_for_display(frame)
                if outputs[1] >= outputs[0]:
                    log.info(
                        "detected class=%s class0=%.4f class1=%.4f",
                        CLASS_NAMES[1],
                        outputs[0].item(),
                        outputs[1].item(),
                    )
                    # paint a small red square so the window shows it too
                    display[0:INDICATOR_SIZE, 0:INDICATOR_SIZE] = INDICATOR_COLOR

                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()
