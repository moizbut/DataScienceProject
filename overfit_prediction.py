import tensorflow as tf
import numpy as np
import os
import logging

image_size = (256, 256)
max_seq_length = 4
data_dir = "/Users/moizmac/DataScienceProject/urdu_seq_dataset_256x256"
model_path = os.path.join(data_dir, "urdu_ocr_sequence_model_reduced_overfit.keras")
mapping_file = os.path.join(data_dir, "char_mapping_256x256.npy")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

char_mapping = np.load(mapping_file, allow_pickle=True).item()
num_classes = len(char_mapping) + 1
label_to_char = char_mapping

logger.info("Loading model...")
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'multi_label_loss': lambda y_true, y_pred: tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)}
)

def load_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def predictions_to_text(predictions, label_to_char, num_classes):
    pred_labels = np.argmax(predictions, axis=-1)
    text = ''
    for label in pred_labels[0]:
        if label < num_classes - 1:
            text += label_to_char.get(label, '')
    return text

def predict_image(image_path):
    if not os.path.exists(image_path):
        logger.error(f"Image path {image_path} does not exist.")
        return None
    
    img = load_image(image_path)
    if img is None:
        return None
    
    img = tf.expand_dims(img, axis=0)
    
    logger.info("Making prediction...")
    predictions = model.predict(img)
    
    predicted_text = predictions_to_text(predictions, label_to_char, num_classes)
    return predicted_text

def main():
    while True:
        image_path = input("Enter the image path (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            logger.info("Exiting...")
            break
        
        predicted_text = predict_image(image_path)
        if predicted_text is not None:
            logger.info(f"Predicted text: {predicted_text}")
        else:
            logger.info("Please try another image path.")

if __name__ == "__main__":
    main()