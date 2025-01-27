import asyncio
import cv2
import ffmpeg
import os
from datetime import datetime


# async def record_video_with_location(event: asyncio.Event, output_file: str, location: tuple):

async def record_video_with_location(event: asyncio.Event, output_file: str):
    """
    Grava um vídeo enquanto o evento é alternado (pressionar botão para iniciar/parar).
    
    Args:
        event (asyncio.Event): O evento para alternar o estado de gravação.
        output_file (str): Caminho do arquivo de saída do vídeo.
    """
    recording = False  # Estado inicial: não está gravando

    print("Pronto para alternar gravação com o botão.")
    try:
        while True:
            # Aguarda o evento ser acionado (botão pressionado)
            await event.wait()
            event.clear()  # Reseta o evento após captura

            # Alterna o estado de gravação
            recording = not recording
            if recording:
                print("Iniciando gravação de vídeo...")
                # Aqui você inicia a gravação (ex.: abrir câmera)
            else:
                print("Parando gravação de vídeo...")
                # Aqui você finaliza a gravação (ex.: fechar arquivo/câmera)

            # Simulação de gravação enquanto ativo
            while recording:
                # Verifica se o evento foi acionado novamente
                if event.is_set():
                    event.clear()  # Reseta o evento
                    recording = not recording  # Alterna o estado de gravação
                    if not recording:
                        print("Parando gravação de vídeo...")
                        break

                
                print("Gravando...")  # Substituir por gravação real
                await asyncio.sleep(1)  # Simula o tempo de gravação

    finally:
        print("Encerrando recursos...")
        if recording:
            print("Liberando recursos de gravação.")  # Fechar câmera ou liberar recursos
