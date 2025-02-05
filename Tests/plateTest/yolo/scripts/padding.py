import cv2
import numpy as np
import os
import random
from os.path import splitext, basename
import re

# Function to parse the .txt file

def parse_car_txt(file_path):
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()

    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]

    if 'plate' in data:
        plate_text = data['plate'].replace(' ', '')
        if len(plate_text) == 7:
            data['letters'] = plate_text[:3]
            data['digits'] = plate_text[3:]
        else:
            print(f"Warning: Unexpected plate format in {file_path}")
            data['letters'], data['digits'] = '', ''
    
    return data

# Function to parse character bounding boxes

def parse_char_data(file_path):
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"char \d+: (\d+) (\d+) (\d+) (\d+)", line.strip())
            if match:
                x, y, w, h = map(int, match.groups())
                boxes.append((x, y, w, h))
    return boxes

# Function to crop and label characters

def crop_plate(image_path, txt_path, output_folder_letters, output_folder_digits):
    car_data = parse_car_txt(txt_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    corners = np.array(car_data['corners'], dtype=np.float32)
    width, height = 256, 96
    target_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_rect)
    cropped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))
    char_boxes = parse_char_data(txt_path)

    if len(char_boxes) != 7:
        print(f"Warning: Unexpected number of characters in {txt_path}")
        return

    for i, (x, y, w, h) in enumerate(char_boxes):
        src_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        dst_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), transformation_matrix).reshape(-1, 2)
        x_min, y_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
        x_max, y_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
        char_img = cropped_image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        base_name = splitext(basename(image_path))[0] + f"_char{i}"
        
        if i < 3:  # Letters
            class_label = ord(car_data['letters'][i]) - ord('A')
            output_img_path = os.path.join(output_folder_letters, base_name + ".png")
            output_label_path = os.path.join(output_folder_letters, base_name + ".txt")
        else:  # Digits
            class_label = int(car_data['digits'][i - 3])
            output_img_path = os.path.join(output_folder_digits, base_name + ".png")
            output_label_path = os.path.join(output_folder_digits, base_name + ".txt")
        
        cv2.imwrite(output_img_path, char_img)
        with open(output_label_path, "w") as f:
            f.write(f"{class_label}")
        
        print(f"Saved {output_img_path} with label {class_label}")

# Main execution
output_folder_letters = '../charRec/crop/val/'
output_folder_digits = '../digitRec/dataset/val/'
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolders:
    image_files += sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Randomly select 1/7 of images
image_files = random.sample(image_files, len(image_files) // 7)

for image_path in image_files:
    txt_path = splitext(image_path)[0] + '.txt'
    crop_plate(image_path, txt_path, output_folder_letters, output_folder_digits)
