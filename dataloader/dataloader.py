from loaddata import get_dataloader, select_datasets

DEFAULT_DATASET_TRAIN = ["enwiki_articles_20240320"]
DEFAULT_DATASET_TEST  = ["enwiki_articles_20240320_TEST"]

class MyDataLoader():
    def __init__(self, promptuser=True, batch_size=1, shuffle=True):
        """
            Custom Data Loader class that allows for custom selection of datasets
            
            Inputs:
                promptuser: (boolean) If true, will call on 'select_datasets' to give a popup that allows for user selection of datasets. If false, will use 'DEFAULT_DATASETS'
                batch_size: (int) Number of samples per batch
                shuffle: (boolean) If true, will shuffle the data when creating the dataloader
        """
        if promptuser:
            self.datasets = select_datasets()
        else:
            self.datasets = DEFAULT_DATASET_TRAIN
        
        self.train_dataloader = get_dataloader(selected_datasets=self.datasets, batch_size=batch_size, shuffle=shuffle)
        self.test_dataloader = get_dataloader(selected_datasets=DEFAULT_DATASET_TEST, batch_size=batch_size, shuffle=shuffle)
    
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader
    
    def print_samples(self, num_samples=10):
        """
            Quick function to print a selected number of samples using your dataloader
        """
        samples_printed = 0
        for batch in self.train_dataloader:
            for sample in batch:
                text_data = sample if isinstance(sample, str) else sample[0]

                print(f"Sample #{samples_printed + 1}")
                print(f"Number of chars: {len(text_data)}")
                print("=" * 50)
                print(text_data[:250])
                print("=" * 50)
                print()

                samples_printed += 1
                if samples_printed >= num_samples:
                    break

            if samples_printed >= num_samples:
                break
        
        print("=" * 50)
        print("Test Dataset Samples:")
        print("=" * 50)
        for batch in self.test_dataloader:
            for sample in batch:
                text_data = sample if isinstance(sample, str) else sample[0]

                print(f"Number of chars: {len(text_data)}")
                print("=" * 50)
                print(text_data[:250])
                print("=" * 50)
                print()
                break
            break
