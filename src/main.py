import asyncio
import cv2
import backupDataBase as database
import gpioRisingDetec as gpio
import wifiConnect as wc
import camera


global_out: cv2.VideoWriter = None
global_recording: bool = False
global_frame: cv2.typing.MatLike = None
global_frame_width: int = 0
global_frame_height: int = 0
global_latitude: float = 0
global_longitude: float = 0


async def main():
    stop_event = asyncio.Event()
    wifi_event = asyncio.Event()
    buttom_event = asyncio.Event()

    try:
        await asyncio.gather(
            wc.is_wifi_connected(wifi_event),
            database.backup_if_wifi(wifi_event),
            gpio.async_watch_line_value(stop_event, buttom_event),
            camera.record_video_with_location(buttom_event),
            camera.capture_frame(),
        )
    except KeyboardInterrupt:
        print("\nInterrupção manual detectada. Encerrando...")
    finally:
        stop_event.set()  # Sinaliza para parar a tarefa
        print("Monitoramento encerrado.")

if __name__ == "__main__":
    asyncio.run(main())
