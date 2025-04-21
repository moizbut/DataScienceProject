import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import logging
import Levenshtein as lev
from datetime import datetime
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration for M1 GPU (MPS) with fallback to CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    logger.info("MPS (M1 GPU) is available and will be used.")
else:
    device = torch.device("cpu")
    logger.info("MPS (M1 GPU) is not available. Falling back to CPU.")

# Hyperparameters
BATCH_SIZE = 16  # Reduced for M1 GPU memory constraints
NUM_EPOCHS = 100
LEARNING_RATE = 1.0
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.5
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_CHARS = 231  # Based on your character set (UrduGlyphs.txt)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Paths
DATA_ROOT = "data/UTRSet-Synth"
LABEL_FILE = os.path.join(DATA_ROOT, "labels.txt")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MODEL_SAVE_PATH = "best_urdu_ocr_model.pth"

# Character set
with open("/Users/moizmac/DataScienceProject/data/UTRSet-Real/UrduGlyphs.txt", "r", encoding="utf-8") as file:
    characters = file.read().strip() + " "
CHAR_TO_IDX = {char: idx for idx, char in enumerate(characters)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(characters)}

# Simple augmentation functions
def salt_and_pepper_noise(img, prob=0.01):
    img_array = np.array(img)
    noise = np.random.random(img_array.shape) < prob
    img_array[noise] = 255 if random.random() < 0.5 else 0
    return Image.fromarray(img_array)

# Dataset class
class UrduOCRDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, max_length=100):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.image_paths = []
        self.labels = []
        
        # Load labels from gt.txt
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split on whitespace and take the first part as filename, last part as label
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            img_name = parts[0]
            label = ' '.join(parts[1:])  # Join the rest as the label
            
            # Remove the 'test/' prefix from the image name
            if img_name.startswith('test/'):
                img_name = img_name[len('test/'):]
            
            img_path = os.path.join(image_dir, img_name)
            
            # Check if the image file exists
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Filter out invalid labels
            if len(label) > self.max_length:
                logger.warning(f"Label too long (>{self.max_length}): {label}")
                continue
            if any(char not in CHAR_TO_IDX for char in label):
                logger.warning(f"Label contains out-of-vocabulary characters: {label}")
                continue
                
            self.image_paths.append(img_path)
            self.labels.append(label)
        
        logger.info(f"Loaded {len(self.image_paths)} samples from {image_dir}")
        if len(self.image_paths) == 0:
            raise ValueError("No valid samples found in the dataset. Check image paths and labels.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('L')
        
        # Flip image to match Urdu's right-to-left direction
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply transform here
        if self.transform:
            img = self.transform(img)
        
        # Encode label
        encoded_label = [CHAR_TO_IDX[char] for char in label]
        label_length = len(encoded_label)
        encoded_label = encoded_label + [0] * (self.max_length - label_length)  # Padding
        return img, torch.tensor(encoded_label), label_length, label

# Temporal Dropout
class TemporalDropout(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(TemporalDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training:
            return x
        batch_size, channels, height, width = x.size()
        mask = torch.ones(batch_size, channels, height, 1, device=x.device)
        mask = torch.dropout(mask, self.drop_rate, train=True) / (1 - self.drop_rate)
        mask = mask.expand(-1, -1, -1, width)
        return x * mask

# UTRNetSmall with BatchNorm
class UTRNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(UTRNetSmall, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(IMG_HEIGHT//4, 1))
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)
        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.out_conv(d2)
        return out

# Main model
class UrduOCRModel(nn.Module):
    def __init__(self, in_channels=1, feature_channels=32, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS, num_chars=NUM_CHARS, dropout=DROPOUT):
        super(UrduOCRModel, self).__init__()
        self.feature_extractor = UTRNetSmall(in_channels, feature_channels)
        self.temporal_dropout = TemporalDropout(drop_rate=0.2)
        self.sequence_model = nn.LSTM(feature_channels, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.temporal_dropout(features)
        features = features.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.sequence_model(features)
        output = self.fc_out(lstm_out)
        return output

# Data transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: salt_and_pepper_noise(img, prob=0.01) if random.random() < 0.25 else img),
    transforms.RandomRotation(5),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset and perform train-test split
train_dataset = UrduOCRDataset(IMAGE_DIR, LABEL_FILE, transform=train_transform, max_length=100)
val_dataset = UrduOCRDataset(IMAGE_DIR, LABEL_FILE, transform=val_transform, max_length=100)

dataset_size = len(train_dataset)
indices = list(range(dataset_size))
random.shuffle(indices)

train_size = int(TRAIN_SPLIT * dataset_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)

# Initialize model, loss, and optimizer
model = UrduOCRModel().to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

# Training and evaluation loop with MPS fallback
best_cer = float('inf')
cpu_device = torch.device("cpu")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for batch_idx, (images, labels, label_lengths, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.log_softmax(2)
        input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long, device=device)

        try:
            # Try computing loss on MPS
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
        except RuntimeError as e:
            # Fallback to CPU if MPS fails
            logger.warning(f"MPS operation failed: {e}. Falling back to CPU for this operation.")
            outputs = outputs.to(cpu_device)
            labels = labels.to(cpu_device)
            label_lengths = label_lengths.to(cpu_device)
            input_lengths = input_lengths.to(cpu_device)
            criterion = criterion.to(cpu_device)
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            # Move back to MPS for the backward pass
            loss = loss.to(device)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    train_loss /= len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Train Loss: {train_loss:.4f}")
    
    # Validation
    model.eval()
    total_cer = 0
    num_samples = 0
    bad_samples = []
    with torch.no_grad():
        for images, labels, label_lengths, raw_labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            _, preds = outputs.max(2)
            preds = preds.transpose(0, 1)
            
            for pred, gt in zip(preds, raw_labels):
                pred_str = ''.join([IDX_TO_CHAR[idx.item()] for idx in pred if idx.item() != 0]).strip()
                gt_str = gt.strip()
                cer = lev.distance(pred_str, gt_str) / max(len(gt_str), 1)
                total_cer += cer
                num_samples += 1
                
                if cer > 0.5:
                    bad_samples.append((gt_str, pred_str, cer))
    
    avg_cer = total_cer / num_samples
    logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation CER: {avg_cer:.4f}")
    
    if bad_samples:
        logger.info("Samples with high CER:")
        for gt, pred, cer in bad_samples[:5]:
            logger.info(f"GT: {gt}, Pred: {pred}, CER: {cer:.4f}")
    
    if avg_cer < best_cer:
        best_cer = avg_cer
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Saved best model with CER: {best_cer:.4f}")
    
    scheduler.step()

    if best_cer < 0.3:
        logger.info(f"Achieved CER {best_cer:.4f} below target 0.3. Stopping training.")
        break