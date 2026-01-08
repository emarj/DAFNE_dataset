from enum import Enum
from typing import Tuple
import numpy as np
from PIL import Image
import math

class EstimationMethod(Enum):
    DIAGONAL = 'diagonal'

class SolutionSizeEstimator2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __init__(self, mode=EstimationMethod.DIAGONAL) -> None:
        self.min_x = math.inf
        self.min_y = math.inf
        self.max_x = -math.inf
        self.max_y = -math.inf

        self.mode = mode



    def update(self, position: Tuple[int, int, float], img_pil: Image.Image) -> None:
        x, y, angle = position

        if self.mode == EstimationMethod.DIAGONAL:
            # here we compute a quick estimate of the solution size assuming centroid is at center (which is the case in our dataset)
            # we use half-diagonal as radius instead of max distance from centroid for simplicity
            # to be precise we should computer the bounding box of the rotated image around the centroid
            r = 0.5 * math.hypot(img_pil.width, img_pil.height)
            d_x = r
            d_y = r
        else:
            raise ValueError(f"Unknown estimation mode: {self.mode}")

        x_min = x - d_x
        y_min = y - d_y
        x_max = x + d_x
        y_max = y + d_y

        self.min_x = min(self.min_x, x_min)
        self.min_y = min(self.min_y, y_min)
        self.max_x = max(self.max_x, x_max)
        self.max_y = max(self.max_y, y_max)

    def get_size(self) -> Tuple[int, int]:
        width = int(math.ceil(self.max_x - self.min_x))
        height = int(math.ceil(self.max_y - self.min_y))
        return (width, height)


def crop_to_content(sol_img, padding) -> Image.Image:
    bbox = sol_img.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("Reassembled solution is empty")
    
    left, upper, right, lower = bbox

    # Add padding but make sure we don't go out of bounds
    left = max(left - padding, 0)
    upper = max(upper - padding, 0)
    right = min(right + padding, sol_img.width)
    lower = min(lower + padding, sol_img.height)

    return sol_img.crop((left, upper, right, lower))

def centroid_rgba(img):
    a = np.array(img)[:, :, 3]        # alpha channel
    ys, xs = np.where(a > 0)          # foreground pixels
    return xs.mean(), ys.mean()


def center_and_pad_rgba(img: Image.Image) -> Image.Image:
    """
    Centers the RGBA image around its centroid and pads it to a square canvas in such a way that rotations
    around the center do not cause clipping.
    """
    # --- 1. Tight bbox using PIL ---
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Empty image")

    cropped = img.crop(bbox)

    # --- 2. Get opaque pixel coordinates ---
    a = np.array(cropped.split()[-1], dtype=np.float32)
    ys, xs = np.nonzero(a > 0)

    # --- 3. Centroid (alpha-weighted optional) ---
    weights = a[ys, xs]
    cx = np.average(xs, weights=weights)
    cy = np.average(ys, weights=weights)

    # --- 4. MAX DISTANCE ---
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx*dx + dy*dy).max()

    # --- 5. Canvas size from that radius ---
    size = 2 * int(np.ceil(r)) + 1  # make it odd

    # --- 6. Integer placement ---
    C = (size - 1) / 2
    tx = int(np.floor(C - cx))
    ty = int(np.floor(C - cy))

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(cropped, (tx, ty), cropped)

    return canvas

def _convert_to_centroid(position, img_pil, solution_size) -> Tuple[int,int,float]:
    """
    Instead of computing the position using linear algebra and trigonometry, we paste the rotated image on an empty canvas
    and compute the centroid of the resulting image.
    """
    empty_canvas = Image.new("RGBA", solution_size, (0, 0, 0, 0))
    x, y, angle = position
    
    img_pil = img_pil.rotate(angle)
    d_x, d_y = img_pil.width / 2, img_pil.height / 2
    
    x_ = int(round(x - d_x))
    y_ = int(round(y - d_y))

    empty_canvas.paste(img_pil, (x_, y_), img_pil)

    c_x_full, c_y_full  = centroid_rgba(empty_canvas)

    x, y = int(round(c_x_full)), int(round(c_y_full))

    return x,y,angle