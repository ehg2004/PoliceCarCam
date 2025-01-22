import subprocess
import asyncio
import time
import backupDataBase
import wifiConnect
import os
import threading



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

# Função principal
async def main():
    global lastWiFiAcess
    # Inicia a verificação de Wi-Fi em paralelo
    wifi_task = asyncio.create_task(wifiConnect.is_wifi_connected())

    while True:
        if wifiFlag == 1:
            if time.time() - lastWiFiAcess > timeInterval:
                backupDataBase.backup_to_csv(host, database, user, password, query, output_file)
                lastWiFiAcess = time.time()
        await asyncio.sleep(1)

# Ponto de entrada
if __name__ == "__main__":
    asyncio.run(main())