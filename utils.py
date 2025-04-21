import torch
from torch.nn.utils.rnn import pad_sequence

def ocr_collate_fn(batch):
    images = []
    labels = []
    label_lengths = []
    image_paths = []

    # Filter out None samples (e.g., failed image loads)
    valid_batch = [sample for sample in batch if sample is not None]
    if not valid_batch:
        raise ValueError("No valid samples in batch")

    for sample in valid_batch:
        images.append(sample['image'])
        labels.append(sample['label'])
        label_lengths.append(sample['label_length'])
        image_paths.append(sample['image_path'])

    # Stack images
    images = torch.stack(images)  # [batch_size, 1, 128, 1024]

    # Pad labels
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)  # [batch_size, max_label_length]

    # Stack label lengths
    label_lengths = torch.stack(label_lengths)  # [batch_size]

    # Validate label lengths
    max_timesteps = 128  # Expected from model.py
    if (label_lengths > max_timesteps).any():
        print(f"Warning: Label lengths {label_lengths.tolist()} exceed max timesteps ({max_timesteps})")

    return {
        "images": images,
        "labels": labels_padded,
        "label_lengths": label_lengths,
        "image_paths": image_paths  # Optional, remove if unused
    }