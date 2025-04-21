import os
import json
import re

LABEL_FILE = "data/UTRSet-Real/train/gt.txt"
OUTPUT_DIR = "scripts"
CHARSET_FILE = os.path.join(OUTPUT_DIR, "char_to_idx.json")
REVERSE_FILE = os.path.join(OUTPUT_DIR, "idx_to_char.json")

def is_urdu_char(char):
    # Urdu Unicode range: U+0600-U+06FF, plus space and some punctuation
    return (0x0600 <= ord(char) <= 0x06FF) or char in {' ', '،', '۔'}

def extract_charset(label_file=LABEL_FILE):
    charset = set()
    sample_labels = []

    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    with open(label_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or '\t' not in line:
                print(f"Skipping malformed line {line_num}: {line}")
                continue
            try:
                img_path, label = line.split('\t', 1)
                charset.update(label)
                if line_num <= 5:
                    sample_labels.append((img_path, label))
            except Exception as e:
                print(f"Error parsing line {line_num}: {line}, {e}")
                continue

    # Validate charset
    charset = sorted(list(charset))
    non_urdu_chars = [c for c in charset if not is_urdu_char(c)]
    if non_urdu_chars:
        print(f"Warning: Non-Urdu characters found: {non_urdu_chars}")
    if len(charset) < 100 or len(charset) > 200:
        print(f"Warning: Unusual charset size: {len(charset)} (expected ~140-160)")

    # Create mappings
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
    char_to_idx['<blank>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Save mappings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHARSET_FILE, 'w', encoding='utf-8') as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)
    with open(REVERSE_FILE, 'w', encoding='utf-8') as f:
        json.dump(idx_to_char, f, ensure_ascii=False, indent=2)

    # Log details
    print(f"\nCharacter set built successfully!")
    print(f"Total characters (excluding blank): {len(charset)}")
    print(f"Sample characters (first 10): {charset[:10]}")
    print(f"Sample labels (first 5):")
    for img_path, label in sample_labels:
        print(f"  {img_path}: {label}")
    print(f"Saved mappings to:\n- {CHARSET_FILE}\n- {REVERSE_FILE}")

if __name__ == "__main__":
    extract_charset()