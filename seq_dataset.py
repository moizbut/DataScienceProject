import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import pandas as pd
import random
import hashlib

output_dir = "/Users/moizmac/DataScienceProject/urdu_seq_dataset_256x256"
original_dir = "/Users/moizmac/DataScienceProject/urdu_dataset_256x256"

if os.path.abspath(output_dir) == os.path.abspath(original_dir):
    raise ValueError(f"Output directory {output_dir} matches original dataset directory. Change to avoid overwriting.")

vocab_file = "/Users/moizmac/DataScienceProject/urduvocab.txt"
with open(vocab_file, 'r', encoding='utf-8') as f:
    characters = f.read().strip().split()
characters = [char for char in characters if char]
char_to_label = {char: idx for idx, char in enumerate(characters)}
label_to_char = {idx: char for char, idx in char_to_label.items()}

original_label_file = os.path.join(original_dir, "labels.csv")
if os.path.exists(original_label_file):
    original_df = pd.read_csv(original_label_file)
    original_chars = original_df['character'].unique()
    original_labels = original_df['label'].unique()
    for char, label in zip(original_chars, original_labels):
        if char_to_label.get(char) != label:
            raise ValueError(f"Character mapping mismatch for {char}: original label={label}, new label={char_to_label.get(char)}")

with open(vocab_file, 'rb') as f:
    vocab_hash = hashlib.md5(f.read()).hexdigest()
print(f"urduvocab.txt MD5 hash: {vocab_hash}")

font_paths = [
    "/Users/moizmac/DataScienceProject/Alvi Nastaleeq Regular.ttf",
    "/Users/moizmac/DataScienceProject/Jameel Noori Nastaleeq Regular.ttf",
    "/Users/moizmac/DataScienceProject/Faiz Lahori Nastaleeq Regular - [UrduFonts.com].ttf",
]
font_size = 100

image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)

num_images = 30000
image_size = (256, 256)
min_seq_length = 2
max_seq_length = 4

def shear_image(image, shear_factor):
    width, height = image.size
    transform = [1, shear_factor, 0,
                 0, 1, 0,
                 0, 0, 1]
    image = image.transform(image.size, Image.AFFINE, transform, resample=Image.BICUBIC, fillcolor=255)
    return image

def generate_image(sequence, font_path, font_size):
    bg_color = random.randint(200, 255)
    image = Image.new('L', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font {font_path} not found. Using default font.")
        font = ImageFont.load_default()

    text = ' '.join(sequence)
    char_bboxes = [font.getbbox(c) for c in sequence]
    char_widths = [bbox[2] - bbox[0] for bbox in char_bboxes]
    spacing = random.randint(10,30 )
    total_width = sum(char_widths) + spacing * (len(sequence) - 1)
    
    if total_width > image_size[0] - 20:
        scale = (image_size[0] - 20) / total_width
        font_size = int(font_size * scale)
        font = ImageFont.truetype(font_path, font_size)
        char_bboxes = [font.getbbox(c) for c in sequence]
        char_widths = [bbox[2] - bbox[0] for bbox in char_bboxes]
        total_width = sum(char_widths) + spacing * (len(sequence) - 1)
    
    x_start = (image_size[0] - total_width) // 2
    y = (image_size[1] - max([bbox[3] - bbox[1] for bbox in char_bboxes])) // 2
    x = x_start
    text_color = random.randint(0, 50)
    
    for i, c in enumerate(sequence):
        draw.text((x, y), c, font=font, fill=text_color)
        x += char_widths[i] + spacing
    
    angle = random.uniform(-5, 5)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)
    
    image = shear_image(image, random.uniform(-0.1, 0.1))
    
    image_np = np.array(image)
    
    image = Image.fromarray(image_np)
    
    image_np = np.array(image)
    brightness = random.uniform(0.95, 1.05)
    contrast = random.uniform(0.95, 1.05)
    image_np = np.clip((image_np * brightness - 128) * contrast + 128, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_np)

    sequence_labels = [char_to_label[c] for c in sequence]
    return image, True, sequence, sequence_labels

data = []
for i in range(num_images):
    seq_length = random.randint(min_seq_length, max_seq_length)
    sequence = random.choices(characters, k=seq_length)
    
    font_path = random.choice(font_paths)
    
    try:
        image, is_multi_char, sequence, sequence_labels = generate_image(sequence, font_path, font_size)
        image_filename = f"seq_{i}.png"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path)
        data.append({
            'image_path': image_path,
            'sequence': ' '.join(sequence),
            'sequence_labels': sequence_labels,
            'is_multi_char': is_multi_char
        })
    except Exception as e:
        print(f"Error generating image {i}: {e}")
        continue

label_file = os.path.join(output_dir, "labels.csv")
label_df = pd.DataFrame(data)
label_df.to_csv(label_file, index=False)
print(f"Label file saved at {label_file}")

mapping_file = os.path.join(output_dir, "char_mapping_256x256.npy")
np.save(mapping_file, label_to_char)
print(f"Character mapping saved at {mapping_file}")

print(f"Dataset generated at {output_dir} with {len(data)} images.")