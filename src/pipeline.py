import neural_network.index as neural_network
import asyncio
import lcd
import database
import buzzer
import camera
import main as g
import cv2

# TODO: Test detect_plate function with and without the pipeline
async def detect_plate():
    while (1):
        await asyncio.sleep(0.1)
        plate, score = neural_network.license_plate_recognition_pipeline(g.global_frame)
        if score > 0.8:
            print(score, plate)
            cv2.imshow("detect_plate", g.global_frame)
            cv2.waitKey(5000)
            has_log, type, severity = database.get_plate_from_database(plate)
            if (has_log):
                camera.init_recording()
                lcd.escrever_lcd(f"{plate} {severity}", f"* {type}")
                await buzzer.buzzer()
        
        
        