from pathlib import Path
from PIL import Image
from .utils import center_and_pad_rgba, SolutionSizeEstimator2D, crop_to_content


def _reassemble_solution_2d(images, positions, solution_size, centroid_centered) -> Image.Image:

    if not centroid_centered:
        return _reassemble_solution_original_2d(images, positions, solution_size)

    solution_pil = Image.new("RGBA", solution_size, (0, 0, 0, 0))

    for img_pil_original, pos in zip(images, positions):
        x, y, angle = pos

        img_pil = center_and_pad_rgba(img_pil_original)
        img_pil = img_pil.rotate(angle)
        c_x, c_y = img_pil.width / 2, img_pil.height / 2
        

        solution_pil.paste(img_pil, (int(round(x - c_x)), int(round(y - c_y))), img_pil)

    return solution_pil


def _reassemble_solution_original_2d(images, positions, solution_size) -> Image.Image:
    """
    Reference reassembly method that uses the original top-left corner positions.
    """

    solution_pil = Image.new("RGBA", solution_size, (0, 0, 0, 0))

    for img_pil, pos in zip(images, positions):
        x, y, angle = pos
       
        img_pil = img_pil.rotate(angle)

        d_x, d_y = img_pil.width / 2, img_pil.height / 2
        
    
        x_ = int(round(x - d_x))
        y_ = int(round(y - d_y))

        solution_pil.paste(img_pil, (x_, y_), img_pil)
    return solution_pil
     

def reassemble_2d(fragments, puzzle_folder=None, solution_size=None, centroid_centered=True, padding=10) -> Image.Image:
    if puzzle_folder is not None:
        puzzle_folder = Path(puzzle_folder)
    else:
        puzzle_folder = Path('')

    if solution_size is None:
        ssc = SolutionSizeEstimator2D()

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
        if x < 0 or y < 0 and solution_size is not None:
            print("Warning: negative position detected in fragment ", frag.get('idx', 'unknown'))
        positions.append((x,y,angle))
        
        if solution_size is None:
            ssc.update((x,y,angle), img_pil)

    # if solution_size was provided, we are done
    if solution_size is not None:
        return _reassemble_solution_2d(images, positions, solution_size, centroid_centered)

    # otherwise, use the computed estimate, we adjust the positions if needed
    estimated_solution_size = ssc.get_size()
    if ssc.min_x < 0 or ssc.min_y < 0:
        # adjust positions to be non-negative
        for i in range(len(positions)):
            x, y, angle = positions[i]
            positions[i] = (x - ssc.min_x, y - ssc.min_y, angle)

    sol_img = _reassemble_solution_2d(images, positions, estimated_solution_size, centroid_centered)

    # at the end we crop the solution
    sol_img = crop_to_content(sol_img, padding)
        
    return sol_img

