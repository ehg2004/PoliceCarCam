import asyncio
import cv2
from datetime import datetime
import lcd
import main as g
import taglib
import gps


def save_coodinate():
    # Open the MP4 file
    file = taglib.File(g.global_filename)
    # Set metadata fields
    file.tags["title"] =  g.global_latitude.__str__() + " N, " + g.global_longitude.__str__() + " W"
    # Save the changes
    file.save()


def init_recording():
    if g.global_recording:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    g.global_filename = f"video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    g.global_out = cv2.VideoWriter(
        g.global_filename, fourcc, 20.0, (g.global_frame_width, g.global_frame_height)
    )
    g.global_recording = True


async def record_video_with_location(event: asyncio.Event):
    while True:
        await event.wait()
        event.clear()

        if not g.global_recording:
            lcd.escrever_lcd("None Plate     *", "Detected")
            init_recording()
        else:
            print("Recording stopped and saved.")
            lcd.escrever_lcd("None Plate", "Detected")
            g.global_recording = False
            g.global_out.release()
            gps.read_gps_from_uart6()
            save_coodinate()


async def capture_frame():
    cap = cv2.VideoCapture(0)
    g.global_frame_height = int(cap.get(4))
    g.global_frame_width = int(cap.get(3))

    while True:
        await asyncio.sleep(0.01)

        ret, g.global_frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        if g.global_recording:
            g.global_out.write(g.global_frame)
