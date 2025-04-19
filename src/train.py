import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import SimpleCNN
from src.dataset import ADASDataset
import torch.optim as optim
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ADASDataset("data/annotations.csv", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    torch.save(model.state_dict(), f"experiments/checkpoints/epoch_{epoch}.pt")