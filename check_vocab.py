def build_vocab(label_file):
    chars = set()
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            _, label = line.strip().split(maxsplit=1)
            chars.update(label)
    vocab = sorted(chars)
    vocab.append("|") 
    return vocab

vocab = build_vocab("data/UTRSet-Real/train/gt.txt")
print(f"Vocab Size: {len(vocab)}")

def build_vocab(label_file):
    chars = set()
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            _, label = line.strip().split(maxsplit=1)
            chars.update(label)
    vocab = sorted(chars)
    vocab.append("|")
    print("Vocabulary (All chars):", vocab[:157])
    print("Total vocab size:", len(vocab))
    return vocab

# Calling the function
vocab = build_vocab("data/UTRSet-Real/train/gt.txt")