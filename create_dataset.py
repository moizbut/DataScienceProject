import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import pandas as pd
import random


vocab_file = "/Users/moizmac/DataScienceProject/urduvocab.txt"
with open(vocab_file, 'r', encoding='utf-8') as f:
    characters = f.read().strip().split()
characters = [char for char in characters if char]
char_to_label = {char: idx for idx, char in enumerate(characters)}


font_paths = [
    "/Users/moizmac/DataScienceProject/Alvi Nastaleeq Regular.ttf",
    "/Users/moizmac/DataScienceProject/Jameel Noori Nastaleeq Regular.ttf",
    "/Users/moizmac/DataScienceProject/Faiz Lahori Nastaleeq Regular - [UrduFonts.com].ttf",
]
font_sizes = list(range(30, 81, 2))


output_dir = "urdu_dataset_128x128"
image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)


images_per_char = 1000
image_size = (128, 128)

def shear_image(image, shear_factor):
    width, height = image.size
    transform = [1, shear_factor, 0,
                 0, 1, 0,
                 0, 0, 1]
    image = image.transform(image.size, Image.AFFINE, transform, resample=Image.BICUBIC, fillcolor=255)
    return image

def generate_image(char, font_path, font_size):
    bg_color = random.randint(200, 255)
    image = Image.new('L', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font {font_path} not found. Using default font.")
        font = ImageFont.load_default()
    
    
    bbox = font.getbbox(char)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    
    x += random.randint(-20, 20)
    y += random.randint(-20, 20)
    
    
    text_color = random.randint(0, 50)
    
    
    draw.text((x, y), char, font=font, fill=text_color)
    
    
    angle = random.uniform(-5, 5)  
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)
    
    
    shear_factor = random.uniform(-0.1, 0.1)  
    image = shear_image(image, shear_factor)
    
    
    image_np = np.array(image)
    
    
    noise = np.random.normal(0, random.uniform(2, 5), image_np.shape).astype(np.uint8)  # Reduced from 5–15
    image_np = np.clip(image_np + noise, 0, 255)
    
    
    noise_amount = random.uniform(0, 0.1) 
    num_salt = np.ceil(0.01 * image_np.size * noise_amount)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
    image_np[coords[0], coords[1]] = 255
    num_pepper = np.ceil(0.01 * image_np.size * noise_amount)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
    image_np[coords[0], coords[1]] = 0
    
    
    image = Image.fromarray(image_np)
    
    
    if random.random() < 0.3:
        blur_radius = random.uniform(0.3, 1.0)  
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    
    image_np = np.array(image)
    brightness = random.uniform(0.9, 1.1)  
    contrast = random.uniform(0.9, 1.1)
    image_np = np.clip((image_np * brightness - 128) * contrast + 128, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image_np)


data = []
for char in characters:
    print(f"Generating images for character: {char}")
    for i in range(images_per_char):
        font_path = random.choice(font_paths)
        font_size = random.choice(font_sizes)
        
        try:
            
            image = generate_image(char, font_path, font_size)
            
            
            image_filename = f"{char}_{i}.png"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path)
            
            
            data.append({
                'image_path': image_path,
                'character': char,
                'label': char_to_label[char]
            })
        except Exception as e:
            print(f"Error generating image {i} for character {char}: {e}")
            continue


label_df = pd.DataFrame(data)
label_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
print(f"Dataset generated at {output_dir} with {len(data)} images.")