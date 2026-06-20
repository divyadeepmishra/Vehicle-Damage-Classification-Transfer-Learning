import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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


# DATASET
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

# LOAD DATASET
dataset = VehicleDamageDataset(
    csv_file="Dataset/train/train/train.csv",
    image_dir="Dataset/train/train/images",
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# LOAD MODEL
model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    6
)

model.load_state_dict(
    torch.load(
        "vehicle_damage_resnet50.pth",
        map_location=device
    )
)

model = model.to(device)
print("Model Loaded Successfully")

model.eval()

# EVALUATION
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(
            outputs,
            1
        )
        all_labels.extend(
            labels.cpu().numpy()
        )
        all_predictions.extend(
            predictions.cpu().numpy()
        )

# METRICS
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\nValidation Results")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# CLASSIFICATION REPORT
class_names = [
    "Crack",
    "Scratch",
    "Tire Flat",
    "Dent",
    "Glass Shatter",
    "Lamp Broken"
]

print("\nClassification Report")
print("-" * 30)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)

# CONFUSION MATRIX
print("\nConfusion Matrix")
print("-" * 30)

cm = confusion_matrix(
    all_labels,
    all_predictions
)
print(cm)
