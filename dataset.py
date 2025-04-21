import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UrduOCRDataset(Dataset):
    def __init__(self, label_file, image_root, charset_path, transform=None, max_width=1024, max_height=128):
        self.image_root = image_root
        self.transform = transform
        self.max_width = max_width
        self.max_height = max_height

        with open(charset_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)

        self.samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or '\t' not in line:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue
                try:
                    img_rel_path, label = line.split('\t', 1)
                    img_name = os.path.basename(img_rel_path)  # Robust path handling
                    img_path = os.path.join(image_root, img_name)
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found at {img_path}, line {line_num}")
                        continue
                    self.samples.append((img_path, label))
                except Exception as e:
                    print(f"Error processing line {line_num}: {line}, {e}")
                    continue

        if not self.samples:
            raise ValueError("No valid samples found in label_file")
        print(f"Loaded {len(self.samples)} samples")

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.max_height, self.max_width)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_text = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # Skip in collate_fn

        label_seq = []
        for char in label_text:
            if char not in self.char_to_idx:
                raise ValueError(f"Character '{char}' not found in char_to_idx, image: {img_path}")
            label_seq.append(self.char_to_idx[char])
        
        label_tensor = torch.tensor(label_seq, dtype=torch.long)
        label_length = torch.tensor(len(label_seq), dtype=torch.long)
        
        if label_length > 128:  # Match expected timesteps from model.py
            print(f"Warning: Label length {label_length} exceeds max timesteps (128) for {img_path}")

        return {
            "image": image,
            "label": label_tensor,
            "label_length": label_length,
            "image_path": img_path
        }