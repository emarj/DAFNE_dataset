import random

from dafne_dataset import DAFNEMiniDataset
from dafne_dataset.reconstruct import reassemble_2d

def main():

    dataset = DAFNEMiniDataset('.dataset/DAFNE_dataset/', 
                                managed_mode=True,
                                frescos=['*Adoration*'],
                                from_scratch=False,
                                supervised_mode=True)

    print(f"Number of samples in dataset: {len(dataset)}")

    # Select a random index
    
    random_index = random.randint(0, len(dataset) - 1)
    #random_index = 386
    print(f"Selected index: {random_index}")
    data = dataset[random_index]
    #print(sample)

    if dataset.supervised_mode:
        data = data[1]

    print(f"Puzzle: {data['original_puzzle_name']}")
    
    reassemble_img = reassemble_2d(data['fragments'])
    reassemble_img.show()

if __name__ == "__main__":
    main()
