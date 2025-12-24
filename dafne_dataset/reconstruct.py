from pathlib import Path
from typing import Tuple
from PIL import Image
from .utils import centroid_rgba, center_and_pad_rgba

def _convert_to_centroid(position, img_pil, solution_size) -> Tuple[int,int,float]:
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


def _reassemble_solution_2d(images, positions, solution_size) -> Image.Image:

    solution_pil = Image.new("RGBA", solution_size, (0, 0, 0, 0))

    for img_pil_original, pos in zip(images, positions):
        x, y, angle = pos

        img_pil = center_and_pad_rgba(img_pil_original)
        img_pil = img_pil.rotate(angle)
        c_x, c_y = img_pil.width / 2, img_pil.height / 2
        

        solution_pil.paste(img_pil, (int(round(x - c_x)), int(round(y - c_y))), img_pil)

    return solution_pil


def _reassemble_solution_original_2d(images, positions, solution_size) -> Image.Image:

    solution_pil = Image.new("RGBA", solution_size, (0, 0, 0, 0))

    for img_pil, pos in zip(images, positions):
        x, y, angle = pos
       
        img_pil = img_pil.rotate(angle)

        d_x, d_y = img_pil.width / 2, img_pil.height / 2
        
    
        x_ = int(round(x - d_x))
        y_ = int(round(y - d_y))

        solution_pil.paste(img_pil, (x_, y_), img_pil)
    return solution_pil
     

def reassemble_2d(fragments, puzzle_folder=None, solution_size=None) -> Image.Image:
    if puzzle_folder is not None:
        puzzle_folder = Path(puzzle_folder)
    else:
        puzzle_folder = Path('')

    max_x, max_y = 0, 0
    images = []
    positions = []
    for frag in fragments:
        if frag['is_spurious']:
            continue
        
        if 'image' in frag:
            img_pil = frag['image']
        elif 'filename' in frag:
            img_pil = Image.open(puzzle_folder / frag['filename']).convert('RGBA')
        images.append(img_pil)
        
        x,y, angle = frag['position_2d']
        if x < 0 or y < 0:
            print("Warning: negative position detected in fragment ", frag.get('idx', 'unknown'))
        positions.append((x,y,angle))
        
        if solution_size is not None:
            continue
        max_x = max(max_x, x + img_pil.width)
        max_y = max(max_y, y + img_pil.height)

    if solution_size is None:
        solution_size = (max_x, max_y)

    return _reassemble_solution_2d(images, positions, solution_size)