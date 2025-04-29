import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import logging
import pickle
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_size = (128, 128)
batch_size = 32
epochs = 25

data_dir = "/Users/moizmac/DataScienceProject/urdu_dataset_128x128"
label_file = os.path.join(data_dir, "labels.csv")
logs_dir = os.path.join(data_dir, "logs")
checkpoint_dir = os.path.join(data_dir, "checkpoints")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

logger.info("Loading labels.csv...")
df = pd.read_csv(label_file)
logger.info(f"Loaded {len(df)} images, {df['character'].nunique()} unique characters")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label

def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    angle = tf.random.uniform([], -0.087, 0.087)
    image = tfa.image.rotate(image, angle, interpolation='bilinear')
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_dataset(df, batch_size, shuffle=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['label'].values))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_df, batch_size, shuffle=True, augment=True)
test_dataset = create_dataset(test_df, batch_size, shuffle=False, augment=False)

model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 1)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(df['character'].nunique(), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = os.path.join(logs_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
csv_log_file = os.path.join(logs_dir, "training_log.csv")
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras")
history_file = os.path.join(logs_dir, "training_history.pkl")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=False,
        save_weights_only=False,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(csv_log_file, append=False),
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0
    )
]

logger.info("Starting training...")
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

logger.info(f"Saving training history to {history_file}")
with open(history_file, 'wb') as f:
    pickle.dump(history.history, f)

history_json_file = os.path.join(logs_dir, "training_history.json")
with open(history_json_file, 'w') as f:
    json.dump(history.history, f, indent=4)

test_loss, test_acc = model.evaluate(test_dataset)
logger.info(f"Test accuracy: {test_acc:.4f}")
model.save(os.path.join(data_dir, "urdu_ocr_model_128x128.keras"))
char_mapping = dict(zip(df['label'], df['character']))
np.save(os.path.join(data_dir, "char_mapping_128x128.npy"), char_mapping)

logger.info("Training complete. Logs, checkpoints, and history saved.")