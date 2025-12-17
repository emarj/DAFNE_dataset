import bisect
from pathlib import Path
from typing import Union

from PIL import Image

from datman import DataManager
from datman.remote import Remote

from .metadata import retrieve_frescos
        

class DAFNEDataset:

    root : Path
    frescos : list
    include_spurious : bool
    supervised_mode : bool
    managed_mode : bool

    def __init__(self,
                 root : Union[str, Path],
                 frescos : list = [],
                 supervised_mode : bool=False,
                 include_spurious : bool = True,
                 managed_mode: bool = True,
                 from_scratch : bool = False,
                 skip_verify : bool = False) -> None:
        
        
        self.root = Path(root)
        self.managed_mode = managed_mode
        self.supervised_mode = supervised_mode
        self.include_spurious = include_spurious
        
        # iterator state
        self._iter_idx = 0

        self.frescos = frescos

        all_frescos = retrieve_frescos()
        
        if len(self.frescos) == 0:
            # scrape fresco list from website
            self.frescos = list(all_frescos.keys())
        else:
            # check provided fresco list is valid
            for fresco in self.frescos:
                if fresco not in all_frescos:
                    raise ValueError(f"Fresco id '{fresco}' not found in available dataset frescos")

        self.puzzle_list = []
        for fresco in self.frescos:
            if self.managed_mode:
                dm = DataManager(
                    root=self.root,
                    dataset_id=fresco,
                    remote=Remote(
                        url=all_frescos[fresco],
                        filename=fresco + ".zip",
                        root_folder=fresco
                    ),
                    download_folder=self.root / "downloads",
                    extract_subpath='frescos',
                    from_scratch=from_scratch,
                    skip_verify=skip_verify,
                )
            pl = self.load_puzzle(self.root / fresco if not self.managed_mode else dm.data_path)

            self.puzzle_list.append(pl)

        self.cumulative_sizes = self.cumsum(self.puzzle_list)

    # iterator protocol
    def __iter__(self) -> 'DAFNEDataset':
        self._iter_idx = 0
        return self

    def __next__(self) -> Union[dict, tuple]:
        if self._iter_idx >= len(self):
            raise StopIteration
        item = self[self._iter_idx]
        self._iter_idx += 1
        return item
    
    @staticmethod
    def cumsum(sequence) -> list:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    
    @staticmethod
    def load_puzzle(folder) -> list:
        data_path = Path(folder)
        
        if not data_path.exists():
                raise RuntimeError("Dataset path does not exist. Check the specified root folder is correct.")

        puzzle_folders_list = [p for p in data_path.iterdir() if p.is_dir()]
        puzzle_folders_list.sort()

        if len(puzzle_folders_list) == 0:
                raise RuntimeError("No data found in the specified root folder.")
        
        return puzzle_folders_list
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
    
    def _get_idx(self, idx : int) -> tuple:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_metadata(self, idx : int) -> dict:
        puzzle_idx, sample_idx = self._get_idx(idx)
        puzzle_folder = self.puzzle_list[puzzle_idx][sample_idx]
        
        solution_file = puzzle_folder / "fragments.txt"
        solved_fragments = parse_solution(solution_file)

        spurious_file = puzzle_folder / "fragments_s.txt"
        with open(spurious_file, 'r') as file:
            spurious_fragments = [int(line.strip()) for line in file]

        fragments = []

        for f in (puzzle_folder / 'frag_eroded').iterdir():
            if f.suffix == '.png':
                idx = int(f.stem.rsplit('_', 1)[-1])
                fragments.append({
                    'idx': idx,
                    'filepath': str(f),
                    'is_spurious': idx in spurious_fragments,
                    'position_2d': solved_fragments.get(idx, None),
                })
        fragments.sort(key=lambda x: x['idx'])
        
        return {
            'puzzle_name': puzzle_folder.name,
            'fragments': fragments,
            'spurious_fragments': spurious_fragments,
        }

    def __getitem__(self, key : int) -> Union[dict, tuple]:

        data = self.get_metadata(key)

        if not self.supervised_mode:
            return data
        
        ######## SUPERVISED MODE ########
        
        # in this case self.supervised_mode is True
        # we split input x and target
        # x contains in-memory images and few metadata
        # data contains the original metadata dict with the GT


        fragments = []
        for frag in data['fragments']:
            image = Image.open(frag['filepath']).convert('RGBA')

            fragments.append({
                'idx': frag['idx'],
                'image': image,
            })
        
        x = {
            'name': data['puzzle_name'],
            'fragments': fragments,
        }


        return x, data


def parse_solution(solution_path: Union[str, Path]) -> dict:
    data = {}
    with open(solution_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Convert each value to int
            idx, x, y = map(int, parts[:-1])
            angle = float(parts[-1])
            
            data[idx] = (x, y, angle)
    return data