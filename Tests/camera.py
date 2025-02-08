import asyncio
import cv2
from datetime import datetime

out = None
recording = False

frame_width: int = 0  
frame_height: int = 0 

async def record_video_with_location(event: asyncio.Event):
    global recording
    global out
    global frame_width
    global frame_height
    print("b1")
    while True:
        print("b2")
        await event.wait()
        event.clear()

        if not recording:
            print("Recording started.")
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
            recording = True
        else:
            print("Recording stopped and saved.")
            recording = False
            out.release()

async def capture_frame():
    global recording
    global out
    global frame_width
    global frame_height
    print("a1")
    cap = cv2.VideoCapture(0)

    frame_height = int(cap.get(4))
    frame_width = int(cap.get(3))

    cv2.namedWindow("Webcam")

    while True:
        print("a2")
        await asyncio.sleep(0.01)

        ret, frame = cap.read()

        cv2.imshow("Webcam", frame)

        if not ret:
            print('Failed to grab frame')
            break

        if recording:
            out.write(frame)