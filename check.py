import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import argparse

# Paths
DATA_ROOT = "data/data"
LABEL_FILE = os.path.join(DATA_ROOT, "gt.txt")  # Changed to gt.txt
MODEL_PATH = "best_urdu_ocr.pt"
VOCAB_FILE = os.path.join(DATA_ROOT, "/Users/moizmac/DataScienceProject/data/UTRSet-Real/UrduGlyphs.txt")

# Hyperparameters
IMG_HEIGHT = 32
IMG_WIDTH = 128
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

# Device setup
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS for inference")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available, falling back to CPU")

# Step 1: Create character vocabulary
def create_vocab(label_files):
    chars = set()
    for label_file in label_files:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit('\t', 1) if '\t' in line else line.strip().rsplit(' ', 1)
                if len(parts) < 2:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                label = parts[-1]
                chars.update(label)
    chars.add(" ")  # Explicitly add space
    chars = sorted(list(chars))
    chars.append("")  # Blank token for CTC
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char

# Step 2: UTRNet-Small Model
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

# Step 3: Full OCR Model
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

# Step 4: CTC Decoding
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

# Step 5: Image Preprocessing
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

# Step 6: Inference
def predict(model, image_path, idx_to_char):
    model.eval()
    with torch.no_grad():
        img = preprocess_image(image_path)
        img = img.to(DEVICE)
        outputs = model(img)
        print(f"Raw output shape: {outputs.shape}")
        preds = decode_predictions(outputs, idx_to_char)
        return preds[0]

# Main Execution
def main():
    parser = argparse.ArgumentParser(description="Test Urdu OCR model on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    char_to_idx, idx_to_char = create_vocab([LABEL_FILE])
    num_chars = len(char_to_idx)
    print(f"Vocabulary size: {num_chars}")
    
    model = UrduOCRModel(num_chars=num_chars, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT)
    model = model.to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model weights from {MODEL_PATH}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    try:
        prediction = predict(model, args.image, idx_to_char)
        print(f"Predicted text: {prediction}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()