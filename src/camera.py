import asyncio
import cv2
from datetime import datetime


async def record_video_with_location(event: asyncio.Event):
    global global_recording
    global global_out
    global global_frame_width
    global global_frame_height
    while True:
        await event.wait()
        event.clear()

        if not global_recording:
            print("Recording started.")
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            global_out = cv2.VideoWriter(
                filename, fourcc, 20.0, (global_frame_width, global_frame_height)
            )
            global_recording = True
        else:
            print("Recording stopped and saved.")
            global_recording = False
            global_out.release()


async def capture_frame():
    global global_recording
    global global_out
    global global_frame_width
    global global_frame_height
    global global_frame

    cap = cv2.VideoCapture(0)
    global_frame_height = int(cap.get(4))
    global_frame_width = int(cap.get(3))

    while True:
        await asyncio.sleep(0.01)

        ret, global_frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        if global_recording:
            global_out.write(global_frame)
