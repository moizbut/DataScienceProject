import argparse
import os
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_size = (256, 256)
max_seq_length = 4
data_dir = "/Users/moizmac/DataScienceProject/urdu_seq_dataset_256x256"
model_path = os.path.join(data_dir, "urdu_ocr_sequence_model_256x256.keras")
mapping_file = os.path.join(data_dir, "char_mapping_256x256.npy")

def load_char_mapping():
    logger.info("Loading character mapping...")
    char_mapping = np.load(mapping_file, allow_pickle=True).item()
    num_classes = len(char_mapping) + 1
    return char_mapping, num_classes

def preprocess_image(image_path):
    logger.info(f"Loading and preprocessing image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def decode_prediction(pred, label_to_char, blank_class):
    pred = np.argmax(pred, axis=-1)[0]
    seq = [label_to_char.get(p, '?') for p in pred if p != blank_class]
    return ''.join(seq)

def main():
    parser = argparse.ArgumentParser(description="Predict text sequence from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    try:
        label_to_char, num_classes = load_char_mapping()
        blank_class = num_classes - 1

        logger.info(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'multi_label_loss': lambda y_true, y_pred: y_true}
        )

        image = preprocess_image(args.image_path)

        logger.info("Making prediction...")
        prediction = model.predict(image, verbose=0)

        predicted_text = decode_prediction(prediction, label_to_char, blank_class)
        logger.info(f"Predicted text: {predicted_text}")

        print(f"Predicted text: {predicted_text}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main()