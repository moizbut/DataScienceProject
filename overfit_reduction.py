import os
import psutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import legacy
from sklearn.model_selection import train_test_split
import logging
import pickle
import json
from datetime import datetime

image_size = (256, 256)
batch_size = 64
epochs = 15
max_seq_length = 4

data_dir = "/Users/moizmac/DataScienceProject/urdu_seq_dataset_256x256"
label_file = os.path.join(data_dir, "labels.csv")
logs_dir = os.path.join(data_dir, "logs_reduced_overfit")
checkpoint_dir = os.path.join(data_dir, "checkpoints_reduced_overfit")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB")

mapping_file = os.path.join(data_dir, "char_mapping_256x256.npy")
char_mapping = np.load(mapping_file, allow_pickle=True).item()
num_classes = len(char_mapping) + 1
label_to_char = char_mapping
char_to_label = {v: k for k, v in label_to_char.items()}

df = pd.read_csv(label_file)

def pad_sequence(seq):
    seq = eval(seq)
    return seq + [num_classes - 1] * (max_seq_length - len(seq))

df['padded_labels'] = df['sequence_labels'].apply(pad_sequence)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label

def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_dataset(df, batch_size, shuffle=True, augment=False):
    labels = np.stack(df['padded_labels'].values)
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_df, batch_size, augment=True)
test_dataset = create_dataset(test_df, batch_size, augment=False)

model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 1)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(max_seq_length * num_classes),
    layers.Reshape((max_seq_length, num_classes)),
    layers.Softmax(axis=-1)
])

def multi_label_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(losses, axis=-1)

model.compile(
    optimizer=legacy.Adam(learning_rate=0.0005, clipnorm=1.0),
    loss=multi_label_loss,
    metrics=['accuracy']
)

log_dir = os.path.join(logs_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
csv_log_file = os.path.join(logs_dir, "training_log.csv")
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(csv_log_file),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
]

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(data_dir, "urdu_ocr_sequence_model_reduced_overfit.keras"))