import neural_network.index as neural_network
import asyncio
import lcd
import database
import buzzer
import camera

# TODO: Test detect_plate function with and without the pipeline
async def detect_plate():
    global global_frame
    while (1):
        await asyncio.sleep(0.1)
        plate, accuracy = neural_network.license_plate_recognition_pipeline(global_frame)
        if accuracy > 0.5:
            has_log, type, severity = database.get_plate_from_database(plate)
            if (has_log):
                camera.init_recording()
                lcd.escrever_lcd(f"{plate} {severity}", f"* {type}")
                await buzzer.buzzer()
        
        
        