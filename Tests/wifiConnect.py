import subprocess
import asyncio
import time
import backupDataBase

# Configurações
host = "localhost"
database = "postgres"
user = "postgres"
password = "postgres"
query = "SELECT * FROM public.vehicle"
output_file = "backup.csv"
wifiFlag = 0
lastWiFiAcess = 0
timeInterval = 30

# Função assíncrona para verificar conexão Wi-Fi
async def is_wifi_connected():
    global wifiFlag
    while True:
        try:
            # Executa o comando nmcli para verificar conexões ativas
            process = await asyncio.create_subprocess_exec(
                "nmcli", "-t", "-f", "STATE,TYPE", "connection", "show",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            # Verifica se há alguma conexão Wi-Fi ativa
            for line in stdout.decode().splitlines():
                active, conn_type = line.split(":")
                if active == "activated" and conn_type == "802-11-wireless":
                    print("Conectado ao Wi-Fi!")
                    wifiFlag = 1
                    break
            else:
                print("Não há conexão Wi-Fi ativa.")
                wifiFlag = 0

        except Exception as e:
            print(f"Erro inesperado: {e}")
            wifiFlag = 0

        await asyncio.sleep(5)

# Tarefa a ser executada
def execute_task():
    backupDataBase.backup_to_csv(host, database, user, password, query, output_file)

# Função principal
async def main():
    global lastWiFiAcess
    # Inicia a verificação de Wi-Fi em paralelo
    wifi_task = asyncio.create_task(is_wifi_connected())

    while True:
        if wifiFlag == 1:
            if time.time() - lastWiFiAcess > timeInterval:
                execute_task()
                lastWiFiAcess = time.time()
            else:
                print("Aguardando intervalo...\n")
        await asyncio.sleep(1)

# Ponto de entrada
if __name__ == "__main__":
    asyncio.run(main())
