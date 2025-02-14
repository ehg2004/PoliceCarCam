import asyncio
import lcd
import database
import buzzer
import camera

async def detect_plate():
    global global_frame
    while (1):
        await asyncio.sleep(0.1)
        # TODO: detect real plate
        plate = "ABC1D23"
        has_log, type, severity = database.get_plate_from_database(plate)
        if (has_log):
            camera.init_recording()
            lcd.escrever_lcd(f"{plate} {severity}", f"* {type}")
            await buzzer.buzzer()
        
        
        