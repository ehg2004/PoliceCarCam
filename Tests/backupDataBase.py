import pandas as pd
from sqlalchemy import create_engine
import asyncio


def backup_to_csv(host, database, user, password, query, output_file):
    try:
        # Conectar ao banco de dados usando SQLAlchemy
        engine = create_engine(f'postgresql://{user}:{password}@{host}/{database}')
        print("Conexão com o banco de dados estabelecida com sucesso.")

        # Executar a consulta SQL
        df = pd.read_sql_query(query, engine)

        # Exportar os dados para um arquivo CSV
        df.to_csv(output_file, index=False)
        print(f"Backup salvo em: {output_file}")
    
    except Exception as e:
        print(f"Erro ao fazer o backup: {e}")
    
    finally:
        print("Conexão com o banco de dados encerrada.")

async def backup_if_wifi(host, database, user, password, query, output_file, wifi_event):

    timeInterval = 30

    while True:
        await wifi_event.wait()  # Espera até que o Wi-Fi esteja conectado
        try:
            backup_to_csv(host, database, user, password, query, output_file)
            print("Backup realizado com sucesso!")
        except Exception as e:
            print(f"Erro ao realizar backup: {e}")
        await asyncio.sleep(timeInterval)

