from torch.utils.data import Dataset

class P1_CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        """
        dataset1: First dataset (e.g., MNISTM)
        dataset2: Second dataset (e.g., SVHN)
        """
        self.dataset1 = dataset1  # Store the first dataset
        self.dataset2 = dataset2  # Store the second dataset

    def __len__(self):
        """Return the total size of the combined dataset."""
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        """Fetch an item from one of the two datasets based on the index."""
        if idx < len(self.dataset1):
            # If index is within the range of dataset1 (MNISTM)
            image, label = self.dataset1[idx]
            dataset_label = 0  # Indicate that this image is from MNISTM
        else:
            # If index exceeds the length of dataset1, fetch from dataset2 (SVHN)
            image, label = self.dataset2[idx - len(self.dataset1)]
            dataset_label = 1  # Indicate that this image is from SVHN

        # Return the image and a tuple (dataset_label, class_label)
        return image, (dataset_label, label)