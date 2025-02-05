import cv2
import numpy as np
import python.darknet as darknet
from python.darknet import bbox2points
def load_darknet_model(cfg_path, weights_path, data_path):
    # network, class_names, class_colors = darknet.load_network(
    #     cfg_path, data_path, weights_path, batch_size=1
    # )
    # First thing we do is load the neural network.
    network = darknet.load_net_custom(cfg_path.encode("ascii"), weights_path.encode("ascii"), 0, 1)
    class_names = open(data_path).read().splitlines()

    # Generate some random colours to use for each class.  If you don't want the colours to be random,
    # then set the seed to a hard-coded value.
    #random.seed(3)
    class_colors = darknet.class_colors(class_names)
    prediction_threshold = 0.5
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    return network, class_names, class_colors

def detect_objects(image, network, class_names, threshold=0.25):
    width, height = darknet.network_width(network), darknet.network_height(network)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25, hier_thresh=.5, nms=.45)
    darknet.free_image(darknet_image)

    return detections, image_resized

def draw_detections(image, detections, class_colors):
    image_with_boxes = darknet.draw_boxes(detections, image, class_colors)
    return image_with_boxes

def crop_license_plate(image, detections):
    h, w, _ = image.shape
    for label, confidence, bbox in detections:
        if label == "license_plate":
            x, y, bw, bh = map(int, bbox)
            x1, y1 = max(x - bw // 2, 0), max(y - bh // 2, 0)
            x2, y2 = min(x + bw // 2, w), min(y + bh // 2, h)
            return image[y1:y2, x1:x2]
    return None

def extract_top_segments(detections, top_n=7):
    sorted_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)
    return sorted_detections[:top_n]

def order_detections(detections):
    sorted_detections = sorted(detections, key=lambda x: float(x[2][0]), reverse=False)
    return sorted_detections

# def is_motorbike(detections):
#     sorted_detections = sorted(detections, key=lambda x: float(x[2][1]), reverse=True)
#     x=np.array(7)
#     y=np.array(7)
#     bh=np.array(7)
#     for i,(_, _, bbox) in enumerate(top_segments):
#         x_, y_, bw_, bh_ = map(int, bbox)
#         x[i],y[i],bh[i]=x_,y_,bh_
#     mean_ymin=
#     mean_ymaj=np.mean(x[3:])
#     return


def recognize_characters(char_images, network, class_names):
    recognized_chars = []
    resized_chars = []
    for char_image in char_images:
        detections,resized_char = detect_objects(char_image, network, class_names)
        detections=extract_top_segments(detections,1)
        print(detections)
        for label, confidence, bbox in detections:
            if detections:
                recognized_chars.append(label)  # Get the top recognized class
            else:
                recognized_chars.append("?")  # Placeholder for failed detections

    return recognized_chars

def main(image_path, lp_cfg, lp_weights, lp_data, seg_cfg, seg_weights, seg_data, letter_cfg, letter_weights, letter_data, digit_cfg, digit_weights, digit_data):
    # Load LP detection model
    # lp_network, lp_classes, lp_colors = load_darknet_model(lp_cfg, lp_weights, lp_data)

    # Load character segmentation model
    seg_network, seg_classes, seg_colors = load_darknet_model(seg_cfg, seg_weights, seg_data)

    # Load letter recognition model
    letter_network, letter_classes, letter_colors = load_darknet_model(letter_cfg, letter_weights, letter_data)

    # Load digit recognition model
    digit_network, digit_classes, digit_colors = load_darknet_model(digit_cfg, digit_weights, digit_data)


    for i,image_p in enumerate(image_path):
        print(i)
    # Load image
        lp_image = cv2.imread(image_p)
        if lp_image is None:
            print("Error loading image.")
            return
        cv2.imshow("OG", lp_image)
        cv2.waitKey(1)


        # # Step 1: Detect license plate
        # lp_detections, resized_image = detect_objects(image, lp_network, lp_classes)
        
        # # Draw LP detections
        # image_with_boxes = draw_detections(resized_image, lp_detections, lp_colors)
        # cv2.imshow("License Plate Detection", image_with_boxes)
        # cv2.waitKey(0)

        # # Step 2: Crop LP
        # lp_image = crop_license_plate(image, lp_detections)
        # if lp_image is None:
        #     print("No license plate detected.")
        #     return

        # cv2.imshow("Cropped License Plate", lp_image)
        # cv2.waitKey(0)

    # Step 3: Detect characters
        char_detections, resized_lp = detect_objects(lp_image, seg_network, seg_classes)



        # Extract the seven most probable segments
        top_segments = extract_top_segments(char_detections, top_n=7)
        top_segments = order_detections(top_segments)
        # Draw character detections

        orig_h, orig_w, _ = lp_image.shape

        # Get Darknet network input size (resize dimensions)
        darknet_w, darknet_h = darknet.network_width(seg_network), darknet.network_height(seg_network)

        # Compute scaling factors (resized → original)
        scale_x = orig_w / darknet_w
        scale_y = orig_h / darknet_h

        char_images = []
        char_bboxes = []  # To store corrected bounding boxes for later visualization
        letter_images = []
        digit_images = []
        # for i,(_, _, bbox) in enumerate(top_segments):
        #     x, y, bw, bh = map(int, bbox)

        #     # Scale bbox coordinates back to original LP size
        #     x = int(x * scale_x)
        #     y = int(y * scale_y)
        #     bw = int(bw * scale_x)
        #     bh = int(bh * scale_y)

        #     x1, y1 = max(x - bw // 2, 0), max(y - bh // 2, 0)
        #     x2, y2 = min(x + bw // 2, orig_w), min(y + bh // 2, orig_h)

        #     # Validate cropping dimensions
        #     if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h or x1 >= x2 or y1 >= y2:
        #         print(f"Warning: Skipping invalid bbox {x1, y1, x2, y2}")
        #         continue  # Skip invalid crops

        #     cropped_char = (lp_image[y1:y2, x1:x2])#.copy()
        #     char_bboxes.append((x1, y1, x2, y2))  # Save for visualization
        #     cv2.imshow("Character Segmentation", lp_image[y1:y2, x1:x2])
        #     cv2.waitKey(500)
        #     if i < 3:
        #         letter_images.append(cropped_char)
        #     else:
        #         digit_images.append(cropped_char)

            # Crop the individual character images from the LP
        orig_h, orig_w, _ = resized_lp.shape
        char_images = []
        padding=4
        h, w, _ = resized_lp.shape
        for i,(_, _, bbox) in enumerate(top_segments):
            x, y, bw, bh = map(int, bbox)
            x1, y1 = max(x - bw // 2 - padding, 0), max(y - bh // 2 - padding, 0)
            x2, y2 = min(x + bw // 2 + padding, w), min(y + bh // 2 + padding, h)
            # Validate cropping dimensions
            if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h or x1 >= x2 or y1 >= y2:
                print(f"Warning: Skipping invalid bbox {x1, y1, x2, y2}")
                continue  # Skip invalid crops

            cropped_char=(resized_lp[y1:y2, x1:x2].copy())
            # cv2.imshow("Character Segmentation", cropped_char)
            # cv2.waitKey(500)
            if i < 3:
                letter_images.append(cropped_char)
            else:
                digit_images.append(cropped_char)



        # Step 4: Recognize the letters and digits (Separate them)
        # First three for letters, last four for digits
        recognized_letters = recognize_characters(letter_images, letter_network, letter_classes)
        recognized_digits = recognize_characters(digit_images, digit_network, digit_classes)
        # Step 5: Draw recognized characters on original image
        y_offset = 20
        recognized_text = "".join(recognized_letters) + "".join(recognized_digits)
        resized_lp_copy=resized_lp.copy()
        lp_with_chars = draw_detections(resized_lp, top_segments, seg_colors)
        cv2.imshow("Character Segmentation", lp_with_chars)
        cv2.waitKey(100    )


        for i, char in enumerate(recognized_text):
            cv2.putText(resized_lp_copy, char, (50 + (i * 30), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show final image with bounding boxes for letters and digits
        cv2.imshow("Final Results", resized_lp_copy)
        cv2.waitKey(100)

        # print("Detected Letters:", letters)
        # print("Detected Digits:", digits)

        # Close all windows
        input("Press Enter to continue...")
    print('end')
    cv2.destroyAllWindows()
    darknet.free_network_ptr(letter_network)
    darknet.free_network_ptr(digit_network)
    darknet.free_network_ptr(seg_network)



    # Close all windows
    # cv2.destroyAllWindows()
import os
import random
if __name__ == "__main__":
    random.seed(3)
        # Get all image paths in the directory
    # image_folder = 'output_crops'
    image_folder = "../yolo/characterSeg/dataset/val/"
    images=sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = random.sample(images, len(images) // 7)

    path="./../yolo/characterSeg/"
    main(
        image_path=images,
        lp_cfg='',
        lp_weights='',
        lp_data='',
        seg_cfg="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/characterSeg/yolo-v3mod/yolov3-tiny.cfg",
        seg_weights="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/characterSeg/backup-01-29/yolov3-tiny_best.weights",
        seg_data="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/characterSeg/classes.names",
        letter_cfg="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/charRec/yolo-v3mod/yolov3-tiny.cfg",
        letter_weights="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/charRec/backup/yolov3-tiny_best.weights",
        letter_data="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/charRec/classes.names",
        digit_cfg="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/digitRec/model/bbd-tiny-PRN.cfg",
        digit_weights="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/digitRec/backup/bbd-tiny-PRN_best.weights",
        digit_data="/home/ehg2004/PoliceCarCam/Tests/plateTest/yolo/digitRec/classes.names"

    
    
    
    )
# classes=1
# train=characterSeg/train.txt
# valid=characterSeg/valid.txt
# names=characterSeg/classes.names
# backup=characterSeg/backup/