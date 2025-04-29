import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

# Check for MPS (Apple Silicon GPU) availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# CNN Model for Urdu Character Recognition (same as training)
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
            nn.Linear(128 * 4 * 4, 512),  # Adjust based on input image size
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

# Preprocess the input image
def preprocess_image(image_path, img_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Main prediction function
def predict_urdu_character(image_path, model_path, vocab_file):
    # Load vocabulary
    vocab = load_vocabulary(vocab_file)
    num_classes = len(vocab)

    # Initialize model and load weights
    model = UrduCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the image
    image = preprocess_image(image_path).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        predicted_char = vocab[predicted_idx.item()]

    return predicted_char

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict Urdu character from an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default="/Users/moizmac/DataScienceProject/urdu_char_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--vocab_file", type=str, default="/Users/moizmac/DataScienceProject/vocab.txt",
                        help="Path to the vocabulary file")
    args = parser.parse_args()

    # Predict the character
    predicted_char = predict_urdu_character(args.image_path, args.model_path, args.vocab_file)
    print(f"Predicted Urdu character: {predicted_char}")