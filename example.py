from dafne_dataset import DAFNEFrescoDataset

def main():

    dataset = DAFNEFrescoDataset('.dataset/DAFNE_dataset/',
                                managed_mode=True,
                                fresco='affreschi_Tarquinia_Itremusici_1280x853',
                                from_scratch=False,
                                supervised_mode=True)

    print(f"Number of samples in dataset: {len(dataset)}")
    
    sample = dataset[0]
    

if __name__ == "__main__":
    main()
