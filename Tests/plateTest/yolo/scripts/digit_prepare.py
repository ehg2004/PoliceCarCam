import cv2
import numpy as np
import os
from os.path import splitext, basename
import re
import shutil

# Function to parse the data from the .txt file
def parse_car_txt(file_path):
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()

    # Convert the corners to a list of float tuples
    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]

    # Extract the plate text (e.g., "MLS5511")
    if 'plate' in data:
        plate_text = data['plate'].replace(' ', '')
        if len(plate_text) == 7:  # Standard format: 3 letters + 4 digits
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

# Function to crop plate and split characters into two datasets
def crop_plate(image_path, txt_path, output_img_path_letters, output_label_path_letters,
               output_img_path_digits, output_label_path_digits):
    # Parse the .txt file
    car_data = parse_car_txt(txt_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    # Extract corners and convert to NumPy array
    corners = np.array(car_data['corners'], dtype=np.float32)

    # Define the target rectangle (output dimensions)
    width = int(max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3])))
    height = int(max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1])))
    target_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Perspective transformation
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_rect)
    cropped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

    # Resize cropped image to 256x96
    resized_image = cv2.resize(cropped_image, (256, 96), interpolation=cv2.INTER_AREA)

    # Read character bounding boxes and transform them
    char_boxes = parse_char_data(txt_path)
    new_labels_letters = []
    new_labels_digits = []

    if len(char_boxes) != 7:
        print(f"Warning: Unexpected number of characters in {txt_path}")
        return

    # Process characters separately (first 3 are letters, last 4 are digits)
    for i, (x, y, w, h) in enumerate(char_boxes):
        src_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        dst_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), transformation_matrix).reshape(-1, 2)

        # Compute new bounding box after transformation
        x_min, y_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
        x_max, y_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
        new_w, new_h = x_max - x_min, y_max - y_min

        # Normalize coordinates for YOLO format
        x_center = (x_min + new_w / 2) / width
        y_center = (y_min + new_h / 2) / height
        new_w /= width
        new_h /= height

        # Assign class labels based on character type
        if i < 3:  # Letters (first 3)
            class_label = ord(car_data['letters'][i]) - ord('A') # Save the actual letter (A-Z)
            new_labels_letters.append(f"{class_label} {x_center:.6f} {y_center:.6f} {new_w:.6f} {new_h:.6f}")
        else:  # Digits (last 4)
            class_label = car_data['digits'][i - 3]  # Save the actual digit (0-9)
            new_labels_digits.append(f"{class_label} {x_center:.6f} {y_center:.6f} {new_w:.6f} {new_h:.6f}")

    # Save images
    cv2.imwrite(output_img_path_letters, resized_image)
    cv2.imwrite(output_img_path_digits, resized_image)

    # Save YOLO labels
    with open(output_label_path_letters, "w") as f:
        f.write("\n".join(new_labels_letters))
    
    with open(output_label_path_digits, "w") as f:
        f.write("\n".join(new_labels_digits))

    print(f"Cropped image for letters saved as: {output_img_path_letters}")
    print(f"Cropped image for digits saved as: {output_img_path_digits}")
    print(f"Labels for letters saved as: {output_label_path_letters}")
    print(f"Labels for digits saved as: {output_label_path_digits}")

# Main script execution
# output_path_base = '../characterSeg/dataset/cropped/test/'
# image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/testing/'

# Main script execution
# output_path_base = '../characterSeg/dataset/val/'
output_path_base_digit = '../digitRec/dataset/val/'
output_path_base_char = '../charRec/dataset/val/'
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

# output_path_base_digit = '../digitRec/dataset/train/'
# output_path_base_char = '../charRec/dataset/train/'
# image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/training/'

# output_path_base_digit = '../digitRec/dataset/test/'
# output_path_base_char = '../charRec/dataset/test/'
# image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/testing/'



image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolders:
    image_files += sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for image_path in image_files:
    txt_path = splitext(image_path)[0] + '.txt'

    base_name = splitext(basename(image_path))[0]
    output_img_path_letters = output_path_base_char + base_name + '.png'
    output_label_path_letters = output_path_base_char + base_name + '.txt'
    output_img_path_digits = output_path_base_digit + base_name + '.png'
    output_label_path_digits = output_path_base_digit + base_name + '.txt'



    crop_plate(image_path, txt_path, output_img_path_letters, output_label_path_letters,
               output_img_path_digits, output_label_path_digits)
