from dataclasses import dataclass
from pathlib import Path
from typing import Union,Tuple, cast
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

        self.supervised_mode = supervised_mode
        self.window_size = window_size

        self.cache = SimpleCache(Path(root) / 'mini_frescos' / f'{window_size}x{window_size}', backend=backend, keep_in_memory=True)

        if len(self.cache) == 0:
            self._prepare()
            print(f"Mini dataset cache created with {len(self.cache)} samples.")


    def _prepare(self) -> None:

        for puzzle in tqdm(self.dataset, desc="Extracting mini puzzles", disable=False):
            puzzle = cast(dict, puzzle) # make type checker happy since __getitem__ returns a Union, but we know it's a dict here
            sol_size = puzzle['solution_size']

            windows = self.extract_mini_puzzles(puzzle, sol_size)

            for window in windows:
                if len(window.fragments_idx) == 0:
                    warnings.warn("Skipping empty mini puzzle")
                    continue

                fragments = [puzzle['fragments'][idx] for idx in window.fragments_idx]

                for frag in fragments:

                    img = Image.open(frag['filename']).convert('RGBA')
                    img = img.crop(img.getchannel("A").getbbox())

                    x,y, angle = frag['position_2d']
                    x_w, y_w = window.position
                    # adjust fragment position relative to window and shift by min_x, min_y
                    frag['position_2d'] = (
                        round(x - x_w,2),
                        round(y - y_w,2),
                        angle
                    )

                puzzle_name = f"{puzzle['puzzle_name']}_window_{self.window_size}x{self.window_size}_{window.idx}"

                mini_puzzle = {
                'fragments': fragments,
                'original_puzzle_name': puzzle['puzzle_name'],
                'puzzle_name': puzzle_name
                }

                self.cache._save(puzzle_name, mini_puzzle)

        self.cache.save_index()

    def extract_mini_puzzles(self, puzzle: dict, sol_size: Tuple[int, int]) -> list:
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

                windows.append(MiniPuzzleWindow(
                    idx=window_idx,
                    position=(i, j),
                    fragments_idx=idx.tolist()
                ))
                window_idx += 1
        
        return windows

    def __len__(self) -> int:
        return len(self.cache)
    
    def __getitem__(self, idx : int) -> Union[dict,Tuple]:
        data = self.cache[idx]

        if not self.supervised_mode:
            return data
        
        ######## SUPERVISED MODE ########
        
        # in this case self.supervised_mode is True
        # we split input x and target
        # x contains in-memory images and few metadata
        # data contains the original metadata dict with the GT


        fragments = []
        for frag in data['fragments']:
            image = Image.open(frag['filename']).convert('RGBA')

            fragments.append({
                'idx': frag['idx'],
                'image': image,
            })
        
        x = {
            'name': data['puzzle_name'],
            'fragments': fragments,
        }

        return x, data

@dataclass
class MiniPuzzleWindow:
    idx: int
    position: Tuple[int, int]
    fragments_idx: list[int]

