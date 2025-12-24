import random

from dafne_dataset import DAFNEDataset
from dafne_dataset.reconstruct import reassemble_2d

def main():

    dataset = DAFNEDataset('.dataset/DAFNE_dataset/', 
                                managed_mode=True,
                                frescos=['*Adoration*'],
                                include_spurious=True,
                                from_scratch=False,
                                supervised_mode=True)

    print(f"Number of samples in dataset: {len(dataset)}")

    # Select a random index
    
    random_index = random.randint(0, len(dataset) - 1)
    #random_index = 3
    
    data = dataset[random_index][1]
    #print(sample)

    num_spurious = len(data['spurious_fragments'])

    print(f"Reassembling sample {random_index} with {len(data['fragments']) - num_spurious} fragments" + (f" (+ {num_spurious} spurious)" if num_spurious > 0 else ""))

    reassemble_img = reassemble_2d(data['fragments'],solution_size=data['solution_size'])
    reassemble_img.show()
    

if __name__ == "__main__":
    main()
