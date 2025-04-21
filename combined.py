import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import Levenshtein
import logging

# Paths for Dataset 1 (preprocessed)
DATA_ROOT = "data/data"
ORIGINAL_LABEL_FILE = os.path.join(DATA_ROOT, "gt.txt")
LABEL_FILE = os.path.join(DATA_ROOT, "gt_clean.txt")
IMAGE_DIR = os.path.join(DATA_ROOT, "test_processed_pre")  # Preprocessed directory

# Paths for Dataset 2 (preprocessed, replace with actual paths)
LABEL_FILE2 = os.path.join(DATA_ROOT, "gt_clean2.txt")
IMAGE_DIR2 = os.path.join(DATA_ROOT, "test_processed2_pre")  # Preprocessed directory

VOCAB_FILE = "/Users/moizmac/DataScienceProject/data/UTRSet-Real/UrduGlyphs.txt"

# Hyperparameters
IMG_HEIGHT = 32
IMG_WIDTH = 128
BATCH_SIZE = 24
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
PATIENCE = 10

# Device setup
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS for training")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available, falling back to CPU")
CPU_DEVICE = torch.device("cpu")

# Step 1: Create character vocabulary
def create_vocab(vocab_file):
    chars = set()
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            char = line.strip()
            if char:
                chars.add(char)
    chars.add(" ")
    print(f"Vocabulary characters before sorting: {chars}")
    chars = sorted(list(chars))
    chars.append("")
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char

# Step 2: Custom Dataset (No Preprocessing)
class UrduOCRDataset(Dataset):
    def __init__(self, label_files, image_dirs, char_to_idx):
        self.image_dirs = image_dirs
        self.char_to_idx = char_to_idx
        self.data = []
        self.dataset_sources = []
        for ds_idx, (label_file, image_dir) in enumerate(zip(label_files, image_dirs)):
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().rsplit('\t', 1) if '\t' in line else line.strip().rsplit(' ', 1)
                    if len(parts) < 2:
                        print(f"Skipping invalid line in {label_file}: {line.strip()}")
                        continue
                    img_name, label = parts
                    img_name = img_name.replace("test/", "").strip()
                    img_path = os.path.join(image_dir, img_name)
                    for ext in [".npy"]:  # Assuming .npy tensors
                        test_path = img_path + ext if ext else img_path
                        if os.path.exists(test_path):
                            self.data.append((test_path, label))
                            self.dataset_sources.append(ds_idx)
                            break
                    else:
                        print(f"Image not found for: {img_name} in {image_dir}")
        print("First 5 samples:")
        for i, (path, lbl) in enumerate(self.data[:5]):
            ds_name = "Dataset 1" if self.dataset_sources[i] == 0 else "Dataset 2"
            print(f"  {i+1}. {ds_name}: {path} -> {lbl}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # Load preprocessed tensor
        img = np.load(img_path)  # Shape: (1, 32, 128), values: [-1.0, 1.0]
        img = torch.from_numpy(img).float()
        if img.shape != (1, IMG_HEIGHT, IMG_WIDTH):
            raise ValueError(f"Invalid tensor shape {img.shape} for {img_path}, expected (1, {IMG_HEIGHT}, {IMG_WIDTH})")
        
        label_encoded = []
        for char in label:
            if char in self.char_to_idx:
                label_encoded.append(self.char_to_idx[char])
            else:
                raise ValueError(f"Unknown character '{char}' in label: {label}. Check vocabulary.")
        return img, torch.tensor(label_encoded), len(label_encoded)

# Step 3: Custom Collate Function
def custom_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths

# Step 4: UTRNet-Small Model (UNet-based Multiscale CNN without BatchNorm)
class UTRNetSmall(nn.Module):
    def __init__(self):
        super(UTRNetSmall, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        u4 = self.up4(b)
        d4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(d4)
        u3 = self.up3(d4)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(d3)
        u2 = self.up2(d3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)
        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)
        return d1

# Step 5: Full OCR Model (UTRNet-Small + BiLSTM + CTC)
class UrduOCRModel(nn.Module):
    def __init__(self, num_chars, hidden_size=256, num_layers=2, dropout=0.2):
        super(UrduOCRModel, self).__init__()
        self.cnn = UTRNetSmall()
        self.fc = nn.Linear(32 * 32, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        batch_size = x.size(0)
        c = self.cnn(x)
        c = c.permute(0, 3, 1, 2)
        c = c.reshape(batch_size, c.size(1), -1)
        c = self.fc(c)
        r, _ = self.rnn(c)
        out = self.fc_out(r)
        return out

# Step 6: CTC Decoding
def decode_predictions(preds, idx_to_char):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    batch_size = preds.size(1)
    texts = []
    for i in range(batch_size):
        pred = preds[:, i]
        text = []
        last_char = None
        for idx in pred:
            idx = idx.item()
            if idx != last_char and idx != len(idx_to_char) - 1:
                text.append(idx_to_char.get(idx, ""))
            last_char = idx
        texts.append("".join(text))
    return texts

# Step 7: Training Loop with Early Stopping and Debugging
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, idx_to_char):
    best_cer = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        debug_printed = False
        for batch_idx, (images, labels, label_lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(DEVICE)
            labels = labels.to(CPU_DEVICE)
            label_lengths = label_lengths.to(CPU_DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            batch_size = images.size(0)
            seq_len = outputs.size(1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(CPU_DEVICE)
            if not debug_printed:
                print(f"Batch size: {batch_size}")
                print(f"Outputs shape: {outputs.shape}, device: {outputs.device}")
                print(f"Labels shape: {labels.shape}, device: {labels.device}")
                print(f"Input lengths: {input_lengths.shape}, {input_lengths}, device: {input_lengths.device}")
                print(f"Label lengths: {label_lengths.shape}, {label_lengths}, device: {label_lengths.device}")
                debug_printed = True
            log_probs = outputs.permute(1, 0, 2).log_softmax(2).to(CPU_DEVICE)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step(train_loss)
        
        model.eval()
        val_cer = 0
        val_count = 0
        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths) in enumerate(val_loader):
                images = images.to(DEVICE)
                labels = labels.to(CPU_DEVICE)
                label_lengths = label_lengths.to(CPU_DEVICE)
                outputs = model(images)
                batch_size = images.size(0)
                preds = outputs.permute(1, 0, 2)
                preds = torch.softmax(preds, dim=2)
                preds = torch.argmax(preds, dim=2)
                if batch_idx == 0:
                    print(f"Epoch {epoch+1}, Validation Batch {batch_idx+1}/{len(val_loader)} Raw Predictions (first 5 samples, first 10 timesteps):")
                    for i in range(min(5, batch_size)):
                        pred = preds[:, i][:10].cpu().numpy()
                        decoded = [idx_to_char.get(idx.item(), "<UNK>") for idx in preds[:, i][:10]]
                        print(f"Sample {i+1}: Indices: {pred}, Decoded: {decoded}")
                
                preds = decode_predictions(outputs, idx_to_char)
                if batch_idx == 0:
                    print(f"Epoch {epoch+1}, Validation Batch {batch_idx+1}/{len(val_loader)} Decoded Predictions (first 5 samples):")
                    for i in range(min(5, len(preds))):
                        print(f"Sample {i+1}: {preds[i]}")
                
                if len(preds) != batch_size:
                    logging.warning(f"Prediction length {len(preds)} does not match batch size {batch_size}")
                    continue
                label_start = 0
                true_labels = []
                for length in label_lengths:
                    if label_start + length > labels.size(0):
                        logging.warning(f"Label index out of bounds: start {label_start}, length {length}, labels size {labels.size(0)}")
                        break
                    lbl = labels[label_start:label_start + length]
                    true_labels.append("".join(idx_to_char.get(idx.item(), "?") for idx in lbl))
                    label_start += length
                if len(true_labels) != len(preds):
                    logging.warning(f"True labels length {len(true_labels)} does not match predictions {len(preds)}")
                    continue
                for i, (pred, label) in enumerate(zip(preds, true_labels)):
                    cer = Levenshtein.distance(pred, label) / max(len(label), 1)
                    val_cer += cer
                    val_count += 1
                    if val_count <= 5:
                        logging.info(f"Epoch {epoch+1}, Sample {val_count} - Pred: {pred}, True: {label}, CER: {cer:.4f}")
        
        if val_count > 0:
            val_cer /= val_count
            log_message = f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val CER: {val_cer:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            print(log_message)
            logging.info(log_message)
            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(model.state_dict(), "best_urdu_ocr.pt")
                save_message = f"Saved best model with CER: {best_cer:.4f}"
                print(save_message)
                logging.info(save_message)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in CER for {PATIENCE} epochs)")
                    break
        else:
            log_message = f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val CER: Skipped (no valid samples)"
            print(log_message)
            logging.info(log_message)

# Main Execution
def main():
    logging.basicConfig(
        filename="training_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    
    for label_file in [LABEL_FILE, LABEL_FILE2]:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
    for image_dir in [IMAGE_DIR, IMAGE_DIR2]:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    label_files = [LABEL_FILE, LABEL_FILE2]
    image_dirs = [IMAGE_DIR, IMAGE_DIR2]
    
    char_to_idx, idx_to_char = create_vocab(VOCAB_FILE)
    num_chars = len(char_to_idx)
    print(f"Vocabulary size: {num_chars}")
    
    dataset = UrduOCRDataset(label_files, image_dirs, char_to_idx)
    print(f"Total dataset size: {len(dataset)} samples")
    if len(dataset) == 0:
        raise ValueError("No valid data loaded. Check image paths and label files.")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size < 1 or val_size < 1:
        print("Dataset too small for train-validation split. Using all data for training.")
        train_dataset = dataset
        val_dataset = []
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    ) if val_dataset else None
    
    model = UrduOCRModel(num_chars=num_chars, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT)
    model = model.to(DEVICE)
    
    if os.path.exists("best_urdu_ocr.pt"):
        try:
            model.load_state_dict(torch.load("best_urdu_ocr.pt", map_location=DEVICE))
            print("Loaded best model from previous run")
        except RuntimeError as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting with fresh model")
    
    criterion = nn.CTCLoss(blank=num_chars-1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, idx_to_char)

if __name__ == "__main__":
    main()