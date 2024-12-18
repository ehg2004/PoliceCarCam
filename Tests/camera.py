import cv2 
from datetime import datetime

# Open a connection to the webcam 
cap = cv2.VideoCapture(0)

# Set up window properties
cv2.namedWindow("Webcam")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    cv2.imshow("Webcam", frame)
    
    key = cv2.waitKey(1)
    if key == 32:  # Spacebar pressed
        now = datetime.now()
        # Format the date and time as a string for the filename
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save the captured frame with the timestamp as the filename
        filename = f"photo_{timestamp}.png"
        cv2.imwrite(filename, frame)
    
    # Check if the key pressed is the escape key (ASCII code 27) to exit
    elif key == 27:  # Escape key pressed
        break

cap.release()