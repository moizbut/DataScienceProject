import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os

# Hyperparameters
IMAGE_HEIGHT = 32  # Height of the image
IMAGE_WIDTH = 128  # Width of the image
BATCH_SIZE = 32
EPOCHS = 10

# Define paths
DATA_ROOT = "data/UTRSet-Real"  # Your data directory path
LABEL_FILE = os.path.join(DATA_ROOT, "train/gt.txt")  # Path to the label file
IMAGE_DIR = os.path.join(DATA_ROOT, "train/test")  # Path to the images folder

# Read the gt.txt file and extract image paths and labels
image_paths = []
labels = []

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line.split('\t')) == 2:
            img_path, label = line.split('\t')
            # Combine IMAGE_DIR with the relative image path
            full_img_path = os.path.join(IMAGE_DIR, img_path)
            image_paths.append(full_img_path)  # Store the full image path
            labels.append(label)

# Create char_to_id and id_to_char mappings
characters = sorted(set("".join(labels)))  # Unique characters across all labels
char_to_id = {char: idx + 1 for idx, char in enumerate(characters)}  # Mapping characters to integers
id_to_char = {idx + 1: char for idx, char in enumerate(characters)}  # Reverse mapping for inference

NUM_CLASSES = len(char_to_id) + 1  # Number of characters in our vocabulary (including the blank character for CTC)

# Convert labels to integer sequences using char_to_id mapping
def labels_to_int(labels):
    return [[char_to_id.get(char, 0) for char in label] for label in labels]

# Convert all labels
y_all = labels_to_int(labels)

# Split data into training and validation sets (80-20 split)
train_size = int(0.8 * len(y_all))
y_train = y_all[:train_size]
y_val = y_all[train_size:]

# Model Architecture (CNN + RNN + CTC)
def build_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES):
    input_img = layers.Input(shape=input_shape, name='image_input')
    
    # CNN Layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Reshape to (batch_size, time_steps, features)
    rnn_input = layers.Reshape(target_shape=(IMAGE_WIDTH//4, 128))(x)
    
    # RNN Layers (LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(rnn_input)
    
    # Output layer with CTC loss
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=input_img, outputs=output)
    return model

# CTC Loss Function (for sequence-based output)
def ctc_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length=np.ones(BATCH_SIZE)*IMAGE_WIDTH//4, label_length=np.ones(BATCH_SIZE)*len(max(y_true, key=len))))

# Prepare data for training (make sure to pad sequences)
max_label_length = max(len(label) for label in y_train)  # Use the longest label in y_train for padding
y_train_padded = pad_sequences(y_train, maxlen=max_label_length, padding='post', dtype='int32')
y_val_padded = pad_sequences(y_val, maxlen=max_label_length, padding='post', dtype='int32')

# Convert images to numpy arrays (ensure you have the images paths loaded)
def preprocess_image(img_path):
    print(f"Processing image: {img_path}")  # Debug line to print image path
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize the image
    return img

# Prepare image data for training
X_train = np.array([preprocess_image(img_path) for img_path in image_paths[:len(y_train)]])
X_val = np.array([preprocess_image(img_path) for img_path in image_paths[len(y_train):]])

# Build and compile the model
model = build_model()
model.compile(optimizer=Adam(), loss=ctc_loss)

# Train the model
callbacks = [ModelCheckpoint("ocr_model.h5", monitor="val_loss", save_best_only=True, verbose=1)]

model.fit(
    X_train, 
    y_train_padded, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_data=(X_val, y_val_padded), 
    callbacks=callbacks
)

# Save the final model after training
model.save("final_ocr_model.h5")
