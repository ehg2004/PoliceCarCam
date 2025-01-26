import asyncio
import time
import backupDataBase
import gpioRisingDetec
import gpsCapture

wifi_event = asyncio.Event()

async def is_wifi_connected():
    while True:
        try:
            # Executa o comando nmcli para verificar conexões ativas
            process = await asyncio.create_subprocess_exec(
                "nmcli", "-t", "-f", "STATE,TYPE", "connection", "show",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"Erro no comando nmcli: {stderr.decode()}")
                wifi_event.clear()
                continue

            for line in stdout.decode().splitlines():
                active, conn_type = line.split(":")
                if active == "activated" and conn_type == "802-11-wireless":
                    print("Conectado ao Wi-Fi!")
                    wifi_event.set()
                    break
            else:
                print("Não há conexão Wi-Fi ativa.")
                wifi_event.clear()
        except Exception as e:
            print(f"Erro inesperado: {e}")
            wifi_event.clear()

        await asyncio.sleep(5)

async def backup_if_wifi(host, database, user, password, query, output_file):

    timeInterval = 30

    while True:
        await wifi_event.wait()  # Espera até que o Wi-Fi esteja conectado
        try:
            backupDataBase.backup_to_csv(host, database, user, password, query, output_file)
            print("Backup realizado com sucesso!")
        except Exception as e:
            print(f"Erro ao realizar backup: {e}")
        await asyncio.sleep(timeInterval)

async def main():
    #Configuração do banco
    host = "localhost"
    database = "postgres"
    user = "postgres"
    password = "postgres"
    query = "SELECT * FROM public.vehicle"
    output_file = "backup.csv"

    #Configuração gpio
    chip_path = "/dev/gpiochip1" 
    line_offset = 31
    stop_event = asyncio.Event()

    try:
        await asyncio.gather(
            is_wifi_connected(),
            backup_if_wifi(host, database, user, password, query, output_file),
            gpioRisingDetec.async_watch_line_value(chip_path, line_offset, stop_event, gpsCapture.read_gps_from_uart6)
        )
    except KeyboardInterrupt:
        print("\nInterrupção manual detectada. Encerrando...")
    finally:
        stop_event.set()  # Sinaliza para parar a tarefa
        print("Monitoramento encerrado.")

if __name__ == "__main__":
    asyncio.run(main())
