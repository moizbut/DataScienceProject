import os
import numpy as np
import tensorflow as tf
import argparse
import logging
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_size = (128, 128)
data_dir = "/Users/moizmac/DataScienceProject/urdu_dataset_128x128"
model_path = os.path.join(data_dir, "checkpoints/model_epoch_24_val_acc_0.9858.keras")
char_mapping_path = os.path.join(data_dir, "char_mapping_128x128.npy")

def load_model_and_mapping():
    logger.info("Loading model from %s", model_path)
    if not os.path.exists(model_path):
        logger.error("Model file does not exist: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    logger.info("Loading character mapping from %s", char_mapping_path)
    if not os.path.exists(char_mapping_path):
        logger.error("Character mapping file does not exist: %s", char_mapping_path)
        raise FileNotFoundError(f"Character mapping file not found: {char_mapping_path}")
    char_mapping = np.load(char_mapping_path, allow_pickle=True).item()
    
    return model, char_mapping

def shear_image(image, shear_factor):
    width, height = image.size
    transform = [1, shear_factor, 0,
                 0, 1, 0,
                 0, 0, 1]
    image = image.transform(image.size, Image.AFFINE, transform, resample=Image.BICUBIC, fillcolor=255)
    return image

def load_image(image_path):
    logger.info("Processing image: %s", image_path)
    if not os.path.exists(image_path):
        logger.error("Image file does not exist: %s", image_path)
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path).convert('L')
    logger.info("Original image size: %s", img.size)
    
    img_array = np.array(img, dtype=np.float32)
    logger.info("Image pixel stats: min=%.2f, max=%.2f, mean=%.2f", 
                img_array.min(), img_array.max(), img_array.mean())
    
    thresh_value = img_array.mean() - 20
    thresh = np.where(img_array < thresh_value, 0, 255).astype(np.uint8)
    coords = np.where(thresh == 0)
    if len(coords[0]) == 0:
        logger.warning("No character detected. Using full image.")
        bbox = [0, 0, img_array.shape[1], img_array.shape[0]]
    else:
        x_min, x_max = coords[1].min(), coords[1].max()
        y_min, y_max = coords[0].min(), coords[0].max()
        bbox = [x_min, y_min, x_max, y_max]
    
    padding = 10
    x_min = max(0, bbox[0] - padding)
    y_min = max(0, bbox[1] - padding)
    x_max = min(img_array.shape[1], bbox[2] + padding)
    y_max = min(img_array.shape[0], bbox[3] + padding)
    img = img.crop((x_min, y_min, x_max, y_max))
    logger.info("Cropped character size: %s", img.size)
    
    target_height = np.random.randint(30, 81)
    aspect_ratio = img.size[0] / img.size[1]
    target_width = int(target_height * aspect_ratio)
    if target_width > 100:
        target_width = 100
        target_height = int(target_width / aspect_ratio)
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    logger.info("Resized character size: %s", img.size)
    
    bg_color = np.random.randint(200, 256)
    new_img = Image.new('L', image_size, bg_color)
    x = (image_size[0] - target_width) // 2 + np.random.randint(-20, 21)
    y = (image_size[1] - target_height) // 2 + np.random.randint(-20, 21)
    new_img.paste(img, (x, y))
    img = new_img
    logger.info("Placed on 128x128 canvas with background color: %d", bg_color)
    
    img_array = np.array(img, dtype=np.float32)
    
    text_mask = img_array < (bg_color - 50)
    img_array[text_mask] = img_array[text_mask] * (50 / img_array[text_mask].max())
    img_array[~text_mask] = bg_color
    logger.info("Text color adjusted to 0–50 range")
    
    angle = np.random.uniform(-5, 5)
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)
    img_array = np.array(img, dtype=np.float32)
    logger.info("Applied rotation: %.2f degrees", angle)
    
    shear_factor = np.random.uniform(-0.1, 0.1)
    img = Image.fromarray(img_array.astype(np.uint8))
    img = shear_image(img, shear_factor)
    img_array = np.array(img, dtype=np.float32)
    logger.info("Applied shear: %.2f", shear_factor)
    
    noise_sigma = np.random.uniform(2, 5)
    noise = np.random.normal(0, noise_sigma, img_array.shape).astype(np.float32)
    img_array = np.clip(img_array + noise, 0, 255)
    logger.info("Applied Gaussian noise with sigma: %.2f", noise_sigma)
    
    noise_amount = np.random.uniform(0, 0.1)
    num_salt = int(0.01 * img_array.size * noise_amount)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape]
    img_array[coords[0], coords[1]] = 255
    num_pepper = int(0.01 * img_array.size * noise_amount)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape]
    img_array[coords[0], coords[1]] = 0
    logger.info("Applied salt-and-pepper noise: %.2f%%", noise_amount)
    
    if np.random.random() < 0.3:
        blur_radius = np.random.uniform(0.3, 1.0)
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img_array = np.array(img, dtype=np.float32)
        logger.info("Applied Gaussian blur with radius: %.2f", blur_radius)
    
    brightness = np.random.uniform(0.9, 1.1)
    contrast = np.random.uniform(0.9, 1.1)
    img_array = np.clip((img_array * brightness - 128) * contrast + 128, 0, 255)
    logger.info("Applied brightness: %.2f, contrast: %.2f", brightness, contrast)
    
    img_array_normalized = img_array / 255.0
    
    if np.any(np.isnan(img_array_normalized)) or np.any(np.isinf(img_array_normalized)):
        logger.error("Invalid values in image array (NaN or Inf)")
        raise ValueError("Image array contains invalid values")
    
    img_array_normalized = np.expand_dims(img_array_normalized, axis=(0, -1))
    logger.info("Final image shape: %s", img_array_normalized.shape)
    
    preprocessed_img = Image.fromarray(img_array.astype(np.uint8))
    
    return img_array_normalized, preprocessed_img

def predict_character(model, image, char_mapping):
    logger.info("Running prediction on image")
    logits = model.predict(image, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    predicted_label = np.argmax(probabilities)
    confidence = probabilities[predicted_label]
    predicted_char = char_mapping.get(predicted_label, "Unknown")
    
    top_k_indices = np.argsort(probabilities)[-5:][::-1]
    top_k_chars = [char_mapping.get(idx, "Unknown") for idx in top_k_indices]
    top_k_confidences = probabilities[top_k_indices]
    logger.info("Top-5 predictions:")
    for char, conf in zip(top_k_chars, top_k_confidences):
        logger.info("  %s: %.4f", char, conf)
    
    return predicted_char, confidence, probabilities

def visualize_image(image_path, preprocessed_image, predicted_char, confidence):
    logger.info("Generating visualization")
    original_img = plt.imread(image_path)
    preprocessed_img = np.array(preprocessed_image)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed_img, cmap='gray')
    plt.title(f"Preprocessed Input (128x128)\nPredicted: {predicted_char} (Confidence: {confidence:.4f})")
    plt.axis('off')
    
    plt.show()
    logger.info("Visualization displayed")

def main():
    parser = argparse.ArgumentParser(description="Predict Urdu character with training data preprocessing.")
    parser.add_argument("image_path", type=str, help="Path to the input image (PNG, JPEG, TIFF, WebP, etc.)")
    args = parser.parse_args()

    logger.info("Starting prediction for image: %s", args.image_path)
    
    if not os.path.exists(args.image_path):
        logger.error("Image path does not exist: %s", args.image_path)
        return

    try:
        model, char_mapping = load_model_and_mapping()
    except Exception as e:
        logger.error("Failed to load model or mapping: %s", str(e))
        return

    try:
        image, preprocessed_image = load_image(args.image_path)
    except Exception as e:
        logger.error("Failed to load or preprocess image: %s", str(e))
        return

    try:
        predicted_char, confidence, probabilities = predict_character(model, image, char_mapping)
        logger.info("Predicted character: %s (Confidence: %.4f)", predicted_char, confidence)
        visualize_image(args.image_path, preprocessed_image, predicted_char, confidence)
    except Exception as e:
        logger.error("Prediction or visualization failed: %s", str(e))
        return

if __name__ == "__main__":
    main()