import re
import os

# Input and output file paths
input_file = "/Users/moizmac/DataScienceProject/data/data/gt.txt"
output_file = "/Users/moizmac/DataScienceProject/data/data/gt_clean.txt"

# Define characters to remove based on UrduGlyphs.txt
remove_chars = re.compile(
    r'[A-Za-z0-9۰-۹.\u064B-\u065F\u0670\u0674\u06D4\u06D5\u06FA-\u06FF'
    r'\u060C\u060D\u061B\u061F\u066A-\u066D\uFD3E\uFD3F'
    r'\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u2018-\u201D]'
)

def preprocess_label(label):
    # Remove unwanted characters
    cleaned_label = remove_chars.sub('', label)
    # Remove extra spaces
    cleaned_label = ' '.join(cleaned_label.split())
    return cleaned_label

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read input file and write cleaned labels
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            if '\t' not in line:
                continue  # Skip malformed lines
            img_path, label = line.strip().rsplit('\t', 1)
            cleaned_label = preprocess_label(label)
            if cleaned_label:  # Only write non-empty labels
                f_out.write(f"{img_path}\t{cleaned_label}\n")
            else:
                print(f"Warning: Empty label after cleaning for image {img_path}")

    print(f"Cleaned labels saved to {output_file}")
    print(f"Processed {len(lines)} lines")

if __name__ == "__main__":
    main()