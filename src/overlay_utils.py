from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class OverlaySpec:
    image: np.ndarray
    x: int = 0
    y: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    alpha: float = 1.0


def load_overlay(path):
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)

    return image


def resize_overlay(overlay, width=None, height=None):
    if overlay is None:
        return None

    if width is None and height is None:
        return overlay

    overlay_height, overlay_width = overlay.shape[:2]

    if width is None:
        scale = float(height) / float(overlay_height)
        width = max(1, int(overlay_width * scale))
    elif height is None:
        scale = float(width) / float(overlay_width)
        height = max(1, int(overlay_height * scale))

    return cv.resize(overlay, (max(1, int(width)), max(1, int(height))), interpolation=cv.INTER_AREA)


def compute_bbox_overlay(face_bbox, scale=1.0, offset=(0, 0)):
    if face_bbox is None:
        return None

    x, y, width, height = face_bbox
    offset_x, offset_y = offset
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    center_x = x + width // 2 + offset_x
    center_y = y + height // 2 + offset_y
    left = int(center_x - scaled_width / 2)
    top = int(center_y - scaled_height / 2)
    return left, top, scaled_width, scaled_height


def _split_overlay_channels(overlay, alpha_scale=1.0):
    if overlay.shape[2] == 4:
        color = overlay[:, :, :3]
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    else:
        color = overlay
        alpha = np.ones(overlay.shape[:2], dtype=np.float32)

    return color.astype(np.float32), np.clip(alpha * float(alpha_scale), 0.0, 1.0)


def blend_overlay(frame, overlay, x, y, width=None, height=None, alpha=1.0):
    if frame is None or overlay is None:
        return frame

    working_overlay = resize_overlay(overlay, width=width, height=height)
    if working_overlay is None:
        return frame

    overlay_color, overlay_alpha = _split_overlay_channels(working_overlay, alpha_scale=alpha)
    frame_height, frame_width = frame.shape[:2]
    overlay_height, overlay_width = overlay_color.shape[:2]

    start_x = max(0, int(x))
    start_y = max(0, int(y))
    end_x = min(frame_width, int(x) + overlay_width)
    end_y = min(frame_height, int(y) + overlay_height)

    if start_x >= end_x or start_y >= end_y:
        return frame

    overlay_start_x = start_x - int(x)
    overlay_start_y = start_y - int(y)
    overlay_end_x = overlay_start_x + (end_x - start_x)
    overlay_end_y = overlay_start_y + (end_y - start_y)

    roi = frame[start_y:end_y, start_x:end_x].astype(np.float32)
    overlay_roi = overlay_color[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]
    alpha_roi = overlay_alpha[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x][:, :, None]

    blended = alpha_roi * overlay_roi + (1.0 - alpha_roi) * roi
    frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
    return frame


def composite_overlays(frame, overlays):
    if frame is None:
        return frame

    for overlay in overlays:
        if overlay is None:
            continue

        blend_overlay(
            frame,
            overlay.image,
            overlay.x,
            overlay.y,
            width=overlay.width,
            height=overlay.height,
            alpha=overlay.alpha,
        )

    return frame


def overlay_on_bbox(frame, overlay, face_bbox, scale=1.0, offset=(0, 0), alpha=1.0):
    bbox = compute_bbox_overlay(face_bbox, scale=scale, offset=offset)
    if bbox is None:
        return frame

    x, y, width, height = bbox
    return blend_overlay(frame, overlay, x, y, width=width, height=height, alpha=alpha)