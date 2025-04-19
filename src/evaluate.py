from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from src.dataset import ADASDataset
from src.model import SimpleCNN
import torch

dataset = ADASDataset("data/annotations.csv", transform=None)
loader = DataLoader(dataset, batch_size=32)

model = SimpleCNN()
model.load_state_dict(torch.load("experiments/checkpoints/epoch_9.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in loader:
        preds = model(inputs).argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print(classification_report(all_labels, all_preds))