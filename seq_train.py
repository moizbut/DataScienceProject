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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_size = (256, 256)
batch_size = 64
epochs = 25
max_seq_length = 4

data_dir = "/Users/moizmac/DataScienceProject/urdu_seq_dataset_256x256"
label_file = os.path.join(data_dir, "labels.csv")
logs_dir = os.path.join(data_dir, "logs")
checkpoint_dir = os.path.join(data_dir, "checkpoints")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

logger.debug("Loading character mapping...")
log_memory_usage()
mapping_file = os.path.join(data_dir, "char_mapping_256x256.npy")
char_mapping = np.load(mapping_file, allow_pickle=True).item()
num_classes = len(char_mapping) + 1
label_to_char = char_mapping
char_to_label = {v: k for k, v in label_to_char.items()}
logger.debug(f"Loaded {num_classes-1} characters + 1 blank class")

logger.info("Loading labels.csv...")
df = pd.read_csv(label_file)
logger.info(f"Loaded {len(df)} images")
log_memory_usage()

def pad_sequence(seq):
    try:
        seq = eval(seq)
        if not all(isinstance(l, int) and 0 <= l < num_classes-1 for l in seq):
            raise ValueError(f"Invalid sequence labels: {seq}")
        return seq + [num_classes-1] * (max_seq_length - len(seq))
    except Exception as e:
        logger.error(f"Error parsing sequence: {seq}, Error: {e}")
        raise

df['padded_labels'] = df['sequence_labels'].apply(pad_sequence)
logger.info(f"Dataset prepared with padded labels (max length: {max_seq_length})")
log_memory_usage()

logger.debug("Splitting dataset...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

def load_image(image_path, label):
    logger.debug(f"Loading image: {image_path}")
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label

def augment_image(image, label):
    logger.debug("Applying augmentation")
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_dataset(df, batch_size, shuffle=True, augment=False):
    logger.debug("Creating dataset...")
    labels = np.stack(df['padded_labels'].values)
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logger.debug("Dataset created")
    log_memory_usage()
    return dataset

train_dataset = create_dataset(train_df, batch_size, shuffle=True, augment=False)
test_dataset = create_dataset(test_df, batch_size, shuffle=False, augment=False)

logger.debug("Building model...")
model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(max_seq_length * num_classes, activation=None),
    layers.Reshape((max_seq_length, num_classes)),
    layers.Softmax(axis=-1)
])

def multi_label_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    return tf.reduce_mean(losses, axis=-1)

logger.debug("Compiling model...")
model.compile(optimizer=legacy.Adam(learning_rate=0.0005, clipnorm=1.0),
              loss=multi_label_loss,
              metrics=['accuracy'])

log_dir = os.path.join(logs_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
csv_log_file = os.path.join(logs_dir, "training_log.csv")
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras")
history_file = os.path.join(logs_dir, "training_history.pkl")

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
try:
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

logger.info(f"Saving training history to {history_file}")
with open(history_file, 'wb') as f:
    pickle.dump(history.history, f)

logger.info(f"Saving training history to {history_file}")
def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

history_json_file = os.path.join(logs_dir, "training_history.json")
with open(history_json_file, 'w') as f:
    json.dump(history.history, f, indent=4, default=convert_to_serializable)

def evaluate_model(dataset, label_to_char):
    logger.debug("Evaluating model...")
    total, correct = 0, 0
    blank_class = num_classes - 1
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=-1)
        labels = labels.numpy()
        for i in range(images.shape[0]):
            pred_seq = [label_to_char.get(p, '?') for p in preds[i] if p != blank_class]
            true_seq = [label_to_char.get(l, '?') for l in labels[i] if l != blank_class]
            if ''.join(pred_seq) == ''.join(true_seq):
                correct += 1
            total += 1
    return 100. * correct / total

test_acc = evaluate_model(test_dataset, label_to_char)
logger.info(f"Test sequence accuracy: {test_acc:.2f}%")

model.save(os.path.join(data_dir, "urdu_ocr_sequence_model_256x256.keras"))
logger.info("Training complete. Model and history saved.")