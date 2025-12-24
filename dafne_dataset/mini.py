from .dataset import DAFNEDataset, parse_solution
import numpy as np


class DAFNEMiniDataset:
    def __init__(self, root, frescos, window_size=200, **kwargs) -> None:
        # forward every argument to DAFNEDataset
        self.dataset = DAFNEDataset(root, frescos, **kwargs)

        self.window_size = window_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        puzzle = self.dataset[idx]

        sol_size = puzzle['solution_size']

        canvas_matrix = np.ones(sol_size, dtype=np.int32) * -1

        # build canvas matrix
        for i, frag in enumerate(puzzle['fragments']):
            if frag['is_spurious']:
                continue
            x, y, _ = frag['position_2d']

            canvas_matrix[int(round(x)), int(round(y))] = i

        windows = self.non_overlapping_window_ids(
            canvas_matrix, (self.window_size, self.window_size)
        )

        mini_puzzles = []
        for window in windows:
            if len(window['objects_idx']) == 0:
                continue

            fragments = []
            for frag_idx in window['objects_idx']:
                frag = puzzle['fragments'][frag_idx]
                frag['position_2d'] = (
                    frag['position_2d'][0] - window['bbox'][0],
                    frag['position_2d'][1] - window['bbox'][1],
                    frag['position_2d'][2]
                )
                fragments.append(frag)

            mini_puzzle = {
            'fragments': fragments,
            'solution_size': (self.window_size, self.window_size),
            'puzzle_name': f"{puzzle['puzzle_name']}_window_{self.window_size}x{self.window_size}_{window['idx']}"
            }

            mini_puzzles.append(mini_puzzle)

        return mini_puzzles

        

    @staticmethod
    def non_overlapping_window_ids(matrix, window_size):
        h, w = window_size
        H, W = matrix.shape
        results = []
        window_idx = 0

        for i in range(0, H - h + 1, h):
            for j in range(0, W - w + 1, w):
                bbox = (i, j, i + h, j + w)
                window = matrix[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                idx = np.unique(window[window >= 0])

                results.append({
                    "idx": window_idx,
                    "bbox": bbox,
                    "objects_idx": idx
                })

                window_idx += 1

        return results
        



