import gpiod
import select
import asyncio
from datetime import timedelta
from gpiod.line import Bias, Edge


def edge_type_str(event):
    if event.event_type is event.Type.RISING_EDGE:
        return "Rising"
    if event.event_type is event.Type.FALLING_EDGE:
        return "Falling"
    return "Unknown"


async def async_watch_line_value(chip_path, line_offset, stop_event, event_handler):
    # Configura o GPIO com debounce e detecção de bordas
    with gpiod.request_lines(
        chip_path,
        consumer="async-watch-line-value",
        config={
            line_offset: gpiod.LineSettings(
                edge_detection=Edge.FALLING,
                bias=Bias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            )
        },
    ) as request:
        loop = asyncio.get_event_loop()
        poll = select.poll()
        poll.register(request.fd, select.POLLIN)

        while not stop_event.is_set():
            # Aguarda eventos do GPIO ou do stop_event
            events = await loop.run_in_executor(None, poll.poll, 1000)  # Timeout de 1000ms
            for fd, _event in events:
                if stop_event.is_set():
                    return
                if fd == request.fd:
                    # Lê e processa eventos de borda
                    for event in request.read_edge_events():
                        print(
                            "offset: {}  type: {:<7}  event #{}".format(
                                event.line_offset, edge_type_str(event), event.line_seqno
                            )
                        )
                        event_handler()
                        #colocar a ação que será realizada, iniciar gravação ou salvar loc


# async def main():
#     # Define um evento para sinalizar parada
#     stop_event = asyncio.Event()

#     try:
#         # Inicia a tarefa assíncrona de monitoramento do GPIO
#         monitor_task = asyncio.create_task(
#             async_watch_line_value("/dev/gpiochip1", 31, stop_event)
#         )

#         # Roda indefinidamente até interrupção manual
#         print("Monitoramento iniciado. Pressione Ctrl+C para encerrar.")
#         while not stop_event.is_set():
#             await asyncio.sleep(1)

#     except KeyboardInterrupt:
#         print("\nInterrupção manual detectada. Encerrando...")
#     finally:
#         stop_event.set()  # Sinaliza para parar a tarefa
#         await monitor_task  # Aguarda o término da tarefa
#         print("Monitoramento encerrado.")


# if __name__ == "__main__":
#     asyncio.run(main())
