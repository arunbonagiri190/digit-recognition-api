from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]).permute((1, 2, 0)).contiguous(), self.y[idx]