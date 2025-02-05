import cv2
import os

input_folder = "../characterSeg/dataset/train/.images/"
output_folder = "../characterSeg/dataset/train/resized/"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".png") or file.endswith(".jpg"):
        img = cv2.imread(os.path.join(input_folder, file))
        img_resized = cv2.resize(img, (256,92))
        cv2.imwrite(os.path.join(output_folder, file), img_resized)
