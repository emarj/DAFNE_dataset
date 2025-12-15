from pathlib import Path
from typing import Union

from PIL import Image

from datman import DataManager


class DAFNEFrescoDataset: 

    def __init__(self,
                 root,
                 fresco,
                 managed_mode=True,
                 supervised_mode=False,
                 from_scratch=False,
                 skip_verify=False) -> None:
        
        
        self.root = Path(root)
        
        self.supervised_mode = supervised_mode

        # iterator state
        self._iter_idx = 0

        self.fresco = fresco
        
        if managed_mode:

            # get remote info
            remote = {
                "url": f"https://zenodo.org/records/15800029/files/{fresco}.zip?download=1",
                "checksum": "PLACEHOLDER_FOR_FRESCO_SHA256",
                "filename": f"{fresco}.zip",
                "folder_name": fresco,
            }
            
            self.datamanager = DataManager(
                root=self.root,
                version_type_str=self.fresco,
                remote=remote,
                download_folder = self.root / 'downloads',
                extract_subpath = 'frescos',
                from_scratch=from_scratch,
                skip_verify=skip_verify,
            )

        ################### Load dataset ###################

        self.data_path = self.datamanager.data_path if managed_mode else self.root
        
        err_msg = "Check the specified root folder is correct. If the error persist, try to recreate the dataset running with from_scratch=True or by deleting the root folder."
        if not self.data_path.exists():
            if managed_mode:
                raise RuntimeError(f"Cannot find data folder. {err_msg}")
            else:
                raise RuntimeError("Dataset path does not exist. Check the specified root folder is correct.")

        self.puzzle_folders_list = [p for p in self.data_path.iterdir() if p.is_dir()]
        self.puzzle_folders_list.sort()

        if len(self.puzzle_folders_list) == 0:
            if managed_mode:
                raise RuntimeError(f"No data found after extraction. The dataset may be corrupted. {err_msg}")
            else:
                raise RuntimeError("No data found in the specified root folder.")

    # iterator protocol
    def __iter__(self) -> 'DAFNEFrescoDataset':
        self._iter_idx = 0
        return self

    def __next__(self) -> Union[dict, tuple]:
        if self._iter_idx >= len(self):
            raise StopIteration
        item = self[self._iter_idx]
        self._iter_idx += 1
        return item
    
    def __len__(self) -> int:
        return len(self.puzzle_folders_list)
    
    def _get_metadata(self, key :int) -> dict:
        
        puzzle_folder = self.puzzle_folders_list[key]
        
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

        data = self._get_metadata(key)

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