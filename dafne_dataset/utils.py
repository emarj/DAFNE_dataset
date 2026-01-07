from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import math

def centroid_rgba(img):
    a = np.array(img)[:, :, 3]        # alpha channel
    ys, xs = np.where(a > 0)          # foreground pixels
    return xs.mean(), ys.mean()


def center_and_pad_rgba(img: Image.Image) -> Image.Image:
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


def make_image_grid(
    images,
    cols=None,
    padding=0,
    bg_color=(0, 0, 0, 0),
    order="row",  # "row" or "col"
    grid_thickness=1
) -> Image.Image:
    """
    Create a grid image from a list of same-sized PIL Images.

    Args:
        images (list[PIL.Image.Image]): Images to place in the grid.
        cols (int, optional): Number of columns.
        padding (int): Internal padding in pixels.
        bg_color (tuple): Background color (RGBA).
        order (str): "row" for row-first, "col" for column-first.
        grid_thickness (int): If >0, draw black grid lines of this thickness on top.

    Returns:
        PIL.Image.Image
    """
    if not images:
        raise ValueError("Image list is empty")

    if order not in {"row", "col"}:
        raise ValueError("order must be 'row' or 'col'")

    img_w = max(img.size[0] for img in images)
    img_h = max(img.size[1] for img in images)

    if cols is None:
        cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    grid_w = cols * img_w + (cols - 1) * padding
    grid_h = rows * img_h + (rows - 1) * padding

    grid = Image.new("RGBA", (grid_w, grid_h), bg_color)

    for i, img in enumerate(images):
        if order == "row":
            row = i // cols
            col = i % cols
        else:  # column-first
            col = i // rows
            row = i % rows

        x = col * (img_w + padding)
        y = row * (img_h + padding)

        grid.paste(img, (x, y))

    if grid_thickness and grid_thickness > 0:

        draw = ImageDraw.Draw(grid)
        half_pad = padding / 2.0

        # vertical separators between columns
        for k in range(1, cols):
            x = int(round((k - 1) * (img_w + padding) + img_w + half_pad))
            draw.line([(x, 0), (x, grid_h)], fill=(0, 0, 0, 255), width=grid_thickness)

        # horizontal separators between rows
        for r in range(1, rows):
            y = int(round((r - 1) * (img_h + padding) + img_h + half_pad))
            draw.line([(0, y), (grid_w, y)], fill=(0, 0, 0, 255), width=grid_thickness)

    return grid