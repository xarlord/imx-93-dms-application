"""Image utilities: crop, resize, color conversion, head ROI extraction."""
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def yuyv_to_rgb(
    yuyv_frame: NDArray[np.uint8],
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Convert a raw YUYV (YUY2) frame to RGB.

    Args:
        yuyv_frame: 1-D or 2-D buffer containing YUYV-encoded pixel data.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An (height, width, 3) uint8 array in RGB channel order.
    """
    return cv2.cvtColor(yuyv_frame.reshape((height, width, 2)), cv2.COLOR_YUV2RGB_YUYV)


def crop_with_padding(
    frame: NDArray[np.uint8],
    bbox: list[float],
    padding_ratio: float = 0.20,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Crop a region from a frame with proportional padding, clamped to frame bounds.

    Args:
        frame: Source image as an (H, W, C) uint8 array.
        bbox: Bounding box as [x1, y1, x2, y2].
        padding_ratio: Fraction of the box width/height to add as padding.

    Returns:
        A (cropped_image, (x1, y1, x2, y2)) tuple where the coordinates
        are the clamped crop rectangle in the original frame.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def resize_for_model(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
) -> tuple[NDArray[np.uint8], tuple[float, int, int]]:
    """Resize an image to model input dimensions preserving aspect ratio.

    The image is scaled to fit within *target_size* and the remaining
    area is padded with black pixels.

    Args:
        image: Source image as an (H, W, C) uint8 array.
        target_size: Model input size as (width, height).

    Returns:
        A (resized_canvas, (scale, pad_left, pad_top)) tuple where
        *resized_canvas* is (target_h, target_w, C) and the second
        element carries the geometric transform parameters needed to
        map coordinates back.
    """
    th, tw = target_size[1], target_size[0]
    h, w = image.shape[:2]

    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2

    canvas = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, (scale, pad_left, pad_top)


def crop_eye_region(
    frame: NDArray[np.uint8],
    eye_landmarks: list[tuple[float, float]],
    padding_ratio: float = 0.70,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Crop an eye region from the frame given its 6 landmarks.

    Args:
        frame: Source image as an (H, W, C) uint8 array.
        eye_landmarks: Six (x, y) landmarks around one eye.
        padding_ratio: Fraction of the eye bounding-box dimensions to add
            as padding around the crop.

    Returns:
        A (crop, (x1, y1, x2, y2)) tuple where the coordinates are the
        clamped crop rectangle in the original frame.
    """
    xs = [p[0] for p in eye_landmarks]
    ys = [p[1] for p in eye_landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    fh, fw = frame.shape[:2]
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(fw, int(x2 + pad_x))
    cy2 = min(fh, int(y2 + pad_y))

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def scale_point_to_crop(
    point: tuple[float, float],
    crop_offset: tuple[int, int],
) -> tuple[float, float]:
    """Map a point from frame coordinates to crop coordinates.

    Args:
        point: The (x, y) point in frame coordinates.
        crop_offset: The (x_offset, y_offset) of the crop origin.

    Returns:
        The (x, y) point relative to the crop origin.
    """
    return (point[0] - crop_offset[0], point[1] - crop_offset[1])


def scale_point_from_crop(
    point: tuple[float, float],
    crop_offset: tuple[int, int],
    scale: float,
) -> tuple[float, float]:
    """Map a point from crop/model coordinates back to frame coordinates.

    Args:
        point: The (x, y) point in crop model coordinates.
        crop_offset: The (x_offset, y_offset) of the crop origin.
        scale: The scaling factor that was applied during resizing.

    Returns:
        The (x, y) point in frame coordinates.
    """
    return (point[0] / scale + crop_offset[0], point[1] / scale + crop_offset[1])
