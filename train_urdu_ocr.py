import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for MPS (Apple Silicon GPU) availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset class for Urdu characters
class UrduDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        self.label_to_idx = {char: idx for idx, char in enumerate(vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['image_path'])
        label = self.data.iloc[idx]['character']  # Updated to match CSV column
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            label_idx = self.label_to_idx[label]
            if self.transform:
                image = self.transform(image)
            return image, label_idx
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Skipping...")
            return None, None

# CNN Model for Urdu Character Recognition
class UrduCNN(nn.Module):
    def __init__(self, num_classes):
        super(UrduCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),  # Updated for 256x256 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Load vocabulary
def load_vocabulary(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab

# Custom collate function to handle None values
def custom_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Main training function
def train_model():
    # Parameters
    csv_file = '/Users/moizmac/DataScienceProject/urdu_ocr_dataset/labels.csv'
    root_dir = '/Users/moizmac/DataScienceProject'  # Updated to parent directory
    vocab_file = '/Users/moizmac/DataScienceProject/vocab.txt'
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    img_size = 256  # Updated to match new image size

    # Load vocabulary
    vocab = load_vocabulary(vocab_file)
    num_classes = len(vocab)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    dataset = UrduDataset(csv_file, root_dir, vocab, transform=transform)

    # Split into train and validation sets
    train_data, val_data = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    val_dataset = torch.utils.data.Subset(dataset, val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Initialize model, loss, and optimizer
    model = UrduCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'urdu_char_model.pth')
    print("Model saved as 'urdu_char_model.pth'")

if __name__ == "__main__":
    train_model()