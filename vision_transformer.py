import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ( Dataset, DataLoader, random_split)
from torchvision import transforms
from torchvision import models
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class VehicleDamageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["filename"]
        label = row["label"] - 1
        image_path = os.path.join(
            self.image_dir,
            filename
        )
        image = Image.open(
            image_path
        ).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = VehicleDamageDataset(
    csv_file="Dataset/train/train/train.csv",
    image_dir="Dataset/train/train/images",
    transform=transform
)

print("Dataset Size:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)

print("Train Size:", len(train_dataset))
print("Validation Size:", len(val_dataset))

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

import timm

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=6
)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
print(f"Trainable Parameters: {trainable:,}")

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.fc.parameters(),lr=0.001)

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="weighted")

recall = recall_score(all_labels, all_predictions, average="weighted")

f1 = f1_score(all_labels, all_predictions, average="weighted")

print("\nValidation Results")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

torch.save(model.state_dict(), "vehicle_damage_ViT.pth")
print("\nModel Saved!")
