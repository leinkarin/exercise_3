import torch


# Create a simple dataset and DataLoader
class Dataset(torch.utils.data.Dataset):
    """
    Any dataset should inherit from torch.utils.data.Dataset and override the __len__ and __getitem__ methods.
    __init__ is optional.
    __len__ should return the size of the dataset.
    __getitem__ should return a tuple (data, label) for the given index.
    """

    def __init__(self, samples, labels, device):
        """
        Creates a data set from the given data
        :param samples: a numpy array with two features
        :param labels: a numpy array with zeros or ones
        :param device:
        """
        self.data = torch.tensor(samples, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
