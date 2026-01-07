import math
from pathlib import Path
from typing import cast
import warnings

from tqdm import tqdm

from .dataset import DAFNEDataset
from datman.cache import SimpleCache, NumpyBackend
from PIL import Image
import numpy as np


class DAFNEMiniDataset:
    def __init__(self, root, frescos, window_size=200, supervised_mode=False, backend=NumpyBackend(), **kwargs) -> None:
        # forward every argument to DAFNEDataset
        self.dataset = DAFNEDataset(root, frescos, supervised_mode=False, include_spurious=False, **kwargs)

        if supervised_mode:
            raise NotImplementedError("Supervised mode is not implemented for DAFNEMiniDataset")

        self.supervised_mode = supervised_mode
        self.window_size = window_size

        self.cache = SimpleCache(Path(root) / '.mini_cache', backend=backend, keep_in_memory=True)
        if len(self.cache) == 0:
            self._prepare()

    def _prepare(self) -> None:

        for puzzle in tqdm(self.dataset, desc="Extracting mini puzzles", disable=False):
            puzzle = cast(dict, puzzle) # make type checker happy since __getitem__ returns a Union, but we know it's a dict here
            sol_size = puzzle['solution_size']

            windows = self.extract_mini_puzzles(puzzle, sol_size)

            for window in windows:
                if len(window['fragments_idx']) == 0:
                    warnings.warn("Skipping empty mini puzzle")
                    continue
                
                min_x, min_y = math.inf, math.inf
                max_x, max_y = -math.inf, -math.inf

                fragments = [puzzle['fragments'][idx] for idx in window['fragments_idx']]

                for frag in fragments:
                    x, y, _ = frag['position_2d']

                    img = Image.open(frag['filename']).convert('RGBA')
                    img = img.crop(img.getchannel("A").getbbox())
                    r = 0.5 * math.hypot(img.width, img.height)
                    min_x = min(min_x, x - r)
                    min_y = min(min_y, y - r)
                    max_x = max(max_x, x + r)
                    max_y = max(max_y, y + r)

                # if min_x or min_y are non-negative, we set them to zero
                min_x = min(0, min_x)
                min_y = min(0, min_y)

                solution_size = (int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y)))

                for frag in fragments:
                    x,y, angle = frag['position_2d']
                    x_w, y_w = window['position']
                    # adjust fragment position relative to window and shift by min_x, min_y
                    frag['position_2d'] = (
                        round(x - x_w,2),
                        round(y - y_w,2),
                        angle
                    )

                puzzle_name = f"{puzzle['puzzle_name']}_window_{self.window_size}x{self.window_size}_{window['idx']}"

                mini_puzzle = {
                'fragments': fragments,
                'solution_size': solution_size,
                'original_puzzle_name': puzzle['puzzle_name'],
                'puzzle_name': puzzle_name
                }

                self.cache[puzzle_name] = mini_puzzle
            

    def extract_mini_puzzles(self, puzzle, sol_size):
        canvas_matrix = np.ones(sol_size, dtype=np.int32) * -1

        # build canvas matrix
        for i, frag in enumerate(puzzle['fragments']):
            if frag['is_spurious']:
                continue
            x, y, _ = frag['position_2d']

            canvas_matrix[int(round(x)), int(round(y))] = i

        h, w = self.window_size, self.window_size
        H, W = canvas_matrix.shape
        window_idx = 0

        windows = []

        for i in range(0, H - h + 1, h):
            for j in range(0, W - w + 1, w):
                window = canvas_matrix[i:i+h, j:j+w]
                idx = np.unique(window[window >= 0])

                windows.append({
                    "idx": window_idx,
                    "position": (i, j),
                    "fragments_idx": idx
                })

                window_idx += 1
        
        return windows

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        return self.cache[idx]
    