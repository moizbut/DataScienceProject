import os
import cv2
import glob

# Paths
input_dir = "/Users/moizmac/DataScienceProject/data/data"
output_dir = "/Users/moizmac/DataScienceProject/data/data/processed"
label_file = "/Users/moizmac/DataScienceProject/data/data/gt_clean.txt"
output_label_file = "/Users/moizmac/DataScienceProject/data/data/gt_clean_processed.txt"

def preprocess_image(image_path, output_path, width=128, height=32):
    # Verify input exists
    if not os.path.exists(image_path):
        print(f"Input image not found: {image_path}")
        return False
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    # Resize
    img = cv2.resize(img, (width, height))
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save as PNG
    success = cv2.imwrite(output_path, img)
    if not success:
        print(f"Failed to save image: {output_path}")
        return False
    print(f"Saved image: {output_path}")
    return True

def main():
    # Verify input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Verify label file
    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read gt_clean.txt
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f if '\t' in line]
    
    # Preprocess images and update labels
    processed_count = 0
    skipped_count = 0
    with open(output_label_file, 'w', encoding='utf-8') as f_out:
        for img_name, label in lines:
            input_path = os.path.join(input_dir, img_name)
            output_name = os.path.splitext(img_name)[0] + '.png'
            output_path = os.path.join(output_dir, output_name)
            print(f"Processing: {input_path} -> {output_path}")
            if preprocess_image(input_path, output_path):
                f_out.write(f"{output_name}\t{label}\n")
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"Processed {processed_count} images, Skipped {skipped_count} images")
    print(f"Updated labels saved to {output_label_file}")

if __name__ == "__main__":
    main()