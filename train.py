import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os

# Enable CPU fallback for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configurable Paths
LABEL_FILE_PATH = '/Users/moizmac/DataScienceProject/data/data/gt.txt'
IMAGE_DIR = 'data/data'
VOCAB_FILE_PATH = '/Users/moizmac/DataScienceProject/data/data/UTRSet-Real/UrduGlyphs.txt'

# Step 1: Load Vocabulary from UrduGlyphs.txt
def load_vocabulary(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return vocab, char_to_idx, idx_to_char

# Step 2: Preprocess Images
def preprocess_image(image_path, target_height=32, debug=False):
    full_image_path = os.path.join(IMAGE_DIR, image_path)
    if debug:
        print(f"Trying to load image: {full_image_path}")
    img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {full_image_path}")
    img = img / 255.0
    h, w = img.shape
    target_width = int(w * target_height / h)
    img = cv2.resize(img, (target_width, target_height))
    return img

# Step 3: Custom Collate Function for Padding
def collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    max_width = max(img.shape[2] for img in imgs)
    padded_imgs = []
    for img in imgs:
        width = img.shape[2]
        if width < max_width:
            padding = torch.zeros((1, 32, max_width - width), dtype=torch.float32)
            img = torch.cat([img, padding], dim=2)
        padded_imgs.append(img)
    imgs = torch.stack(padded_imgs, dim=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return imgs, labels, label_lengths

# Step 4: Custom Dataset Class
class UrduOCRDataset(Dataset):
    def __init__(self, labels_df, char_to_idx, transform=None, debug=False):
        self.labels = labels_df
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.labels.iloc[idx, 0]
        text = self.labels.iloc[idx, 1]
        img = preprocess_image(img_path, debug=self.debug)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        label = torch.tensor(label, dtype=torch.int32)
        return img, label, len(label)

# Step 5: CRNN Model Definition
class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(128 * 8, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).view(batch, width, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Step 6: CTC Decoding for Inference
def decode_predictions(outputs, idx_to_char):
    outputs = outputs.argmax(2).cpu().numpy()
    predictions = []
    for batch in outputs.T:
        pred = []
        prev = -1
        for idx in batch:
            if idx != 0 and idx != prev:
                pred.append(idx_to_char.get(idx, ''))
            prev = idx
        predictions.append(''.join(pred))
    return predictions

# Step 7: Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, idx_to_char):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        for batch_idx, (imgs, labels, label_lengths) in enumerate(train_loader):
            batch_count += 1
            print(f"Epoch {epoch+1}, Processing batch {batch_count}/{len(train_loader)}")
            imgs, labels = imgs.to(device), labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = outputs.log_softmax(2)
            input_lengths = torch.full((imgs.size(0),), outputs.size(1), dtype=torch.long).to(device)

            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels, label_lengths in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(imgs)
                outputs = outputs.log_softmax(2)
                input_lengths = torch.full((imgs.size(0),), outputs.size(1), dtype=torch.long).to(device)

                loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
                total_val_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')

# Step 8: Main Execution
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    epochs = 15
    debug_loading = False  # Set to True to print image loading messages

    # Device setup for MacBook Pro M1 (MPS) or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (GPU) on MacBook Pro M1")
    else:
        device = torch.device("cpu")
        print("MPS not available, falling back to CPU")

    # Load vocabulary
    vocab, char_to_idx, idx_to_char = load_vocabulary(VOCAB_FILE_PATH)
    num_chars = len(vocab)

    # Load dataset
    labels = pd.read_csv(LABEL_FILE_PATH, sep='\t', header=None, names=['image_path', 'text'])
    print(f"Total samples in dataset: {len(labels)}")

    # Optionally reduce dataset size for testing
    # labels = labels[:1000]  # Uncomment to use only the first 1000 samples

    # Split dataset
    train_val, test = train_test_split(labels, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.111, random_state=42)
    print(f"Training samples: {len(train)}, Validation samples: {len(val)}, Test samples: {len(test)}")

    # Create datasets and dataloaders with custom collate_fn
    train_dataset = UrduOCRDataset(train, char_to_idx, debug=debug_loading)
    val_dataset = UrduOCRDataset(val, char_to_idx, debug=debug_loading)
    test_dataset = UrduOCRDataset(test, char_to_idx, debug=debug_loading)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Initialize model, loss, optimizer
    model = CRNN(num_chars).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, idx_to_char)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            predictions = decode_predictions(outputs, idx_to_char)
            print("Sample Predictions:", predictions[:5])
            break

    # Save the model
    torch.save(model.state_dict(), 'urdu_ocr_model.pth')