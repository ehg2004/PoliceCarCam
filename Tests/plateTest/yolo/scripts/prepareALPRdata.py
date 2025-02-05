import cv2
import numpy as np
import os
from os.path 					import splitext, basename

import cv2
import numpy as np
import matplotlib as plt
# Function to parse the data from the .txt file
def parse_car_txt(file_path):
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1  ', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()
    
    # Convert the corners to a list of float tuples
    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]
    return data

def extract_plate_name(txt_path):
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                if key.strip() == 'plate':
                    return value.strip()
    return None
    # return None


# Main function to crop and save the image
def crop_plate(image_path, txt_path, output_path):
    # Parse the .txt file
    car_data = parse_car_txt(txt_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    # Extract corners and plate name
    corners = np.array(car_data['corners'], dtype=np.float32)
    plate_name = car_data['plate'] + '.png'

    # Define the target  (output dimensions)
    width = int(max(np.linalgrectangle.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3])))
    height = int(max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1])))
    target_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Perform perspective transformation
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_rect)
    cropped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

    # Resize the cropped image to 94x24 without distortion
    resized_image = cv2.resize(cropped_image, (416, 416), interpolation=cv2.INTER_AREA)

    # Save the resized image
    output_file = f"{output_path}/{plate_name}"
    cv2.imwrite(output_file, resized_image)
    cv2.imshow("resized", resized_image)
    cv2.waitKey(1)

    print(f"resized '{output_file}'")

import re

def parse_char_data(file_path, img_width, img_height): #char segmentation
    labels = []
    boxes = []
    # Read the input file
    with open(file_path, "r") as f:
        for line in f:
            # Match 'char n:' followed by x, y, width, height
            match_pos = re.match(r"char \d+: (\d+) (\d+) (\d+) (\d+)", line.strip())
            match = re.match(r"char \d+: (\d+) (\d+) (\d+) (\d+)", line.strip())
            if match:
                x, y, w, h = map(int, match.groups())
                # Convert to YOLO format
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                # Class ID is 0 (assuming one class: character)
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                boxes.append((x_min, y_min, x_max, y_max))

    return boxes,labels

# Function to visualize bounding boxes
def visualize_boxes(image_file, boxes):
    # Load the image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

    # Draw each bounding box
    for (x_min, y_min, x_max, y_max) in boxes:
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box
    cv2.imshow("cropped", image)
    cv2.waitKey(1)

    
    # Plot the image with bounding boxes
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()

    # # Optionally save the output
    # if output_path:
    #     os.makedirs(output_path, exist_ok=True)
    #     output_file = os.path.join(output_path, os.path.basename(image_file))
    #     plt.savefig(output_file, bbox_inches="tight")
    #     print(f"Saved visualization to {output_file}")



# Specify the paths to the image, the .txt file, and the output directory
# image_path = 'car.png'
# txt_path = 'car.txt'
output_path_base = '../caracterSeg/dataset/val/'  # Replace with your desired output directory
# image_folder = '../yolo/characterSeg/train'
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

# bname = splitext(basename(img_path))[0]
import shutil



image_subfolder = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolder:
    sorted_=sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_files=image_files+sorted_
for image_path in image_files:
    txt_path= splitext((image_path))[0] + '.txt'
    output_path= output_path_base + 'labels/'+ splitext(basename(image_path))[0] + '.txt'
    # crop_plate(image_path, txt_path, output_path)
    boxes,yolo_labels = parse_char_data(txt_path, 1920, 1080)
    with open(output_path, "a+") as f:
        f.write("\n".join(yolo_labels))
    # print(image_path)
    # visualize_boxes(image_path,boxes)

    output_path= output_path_base + "images/"+ splitext(basename(image_path))[0] + '.png'
    shutil.copyfile(image_path, output_path)

    # boxes,yolo_labels = parse_char_data(txt_path, 1920, 1080)
    # with open(output_path, "a+") as f:
        # f.write("\n".join(yolo_labels))
    print(image_path)
    # visualize_boxes(image_path,boxes)

#     image_path = str(sorted_[0])
#     print(image_path)
#     txt_path= str(splitext(image_path)[0]) + str('.txt')
#     print(extract_plate_name(txt_path))
# #     crop_plate(image_path, txt_path, output_path)

