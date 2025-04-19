from torch.utils.data import Dataset
import cv2
import pandas as pd
import torch

class ADASDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image = cv2.imread(row['image_path'])
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label