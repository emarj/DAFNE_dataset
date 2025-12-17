from dafne_dataset import DAFNEDataset

def main():

    dataset = DAFNEDataset('.dataset/DAFNE_dataset/', 
                                managed_mode=True,
                                frescos=['Giotto_AdorationdsdsOfTheMagi_1000x1008'],
                                from_scratch=False,
                                supervised_mode=True)

    print(f"Number of samples in dataset: {len(dataset)}")
    
    sample = dataset[0]
    breakpoint()
    

if __name__ == "__main__":
    main()
