import torch
from PIL import Image
import torchvision.transforms as transforms
import json

from train_ocr1 import CRNN, decode_output  # Use your actual model and decoder

# Load character mappings
with open('char_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)
id_to_char = {int(k): v for k, v in mappings['id_to_char'].items()}

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_path = '/Users/moizmac/DataScienceProject/crnn_urdu_best.pth'
model = CRNN(num_chars=len(id_to_char)).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Image preprocessing (match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 512)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and preprocess the image
img_path = 'tt.jpg'  # Your test image
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # Shape: [1, 1, 32, 512]

# Inference
with torch.no_grad():
    output = model(image)  # [batch, seq_len, num_classes]
    decoded_texts = decode_output(output, id_to_char)
    decoded_text = decoded_texts[0]  # Since batch size is 1

print("Predicted text:", decoded_text)
