from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect
import asyncio
import sqlite3

def backup_to_sqlite():
    host = "localhost:5432"
    database = "postgres"
    user = "postgres"
    password = "postgres"
    sqlite_db = '../Database/local.db'
    try:
        # Connect to the SQLite database and get the date of the last update
        sqlite_conn = sqlite3.connect(sqlite_db)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT MAX(updated_at) FROM vehicle_log")
        last_update_date = cursor.fetchone()[0]
        cursor.close()
        
        # Connect to the PostgreSQL database using SQLAlchemy
        engine = create_engine(f'postgresql://{user}:{password}@{host}/{database}')
        print("Conexão com o banco de dados estabelecida com sucesso.")
        
        # Read tables from PostgreSQL database into Pandas DataFrames
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        dataframes = {}
        for table in table_names:
            if last_update_date:
                query = f"SELECT * FROM {table} WHERE updated_at > '{last_update_date}'"
            else:
                query = f"SELECT * FROM {table}"
            dataframes[table] = pd.read_sql_query(query, engine)
        
        # Write DataFrames to SQLite database
        for table, df in dataframes.items():
            # Insert new rows
            for index, row in df.iterrows():
                if not last_update_date or (last_update_date and row['created_at'] >= datetime.strptime(last_update_date, "%Y-%m-%d").date()):
                    print(f"Inserting row {index} into table {table}")
                    insert_query = f"""
                    INSERT INTO {table} ({', '.join(df.columns)})
                    VALUES ({', '.join(['?' for _ in df.columns])})
                    """
                    cursor = sqlite_conn.cursor()
                    cursor.execute(insert_query, tuple(row))
                    sqlite_conn.commit()
                    cursor.close()
            # Update existing rows
            for index, row in df.iterrows():
                update_query = f"""
                UPDATE {table}
                SET {', '.join([f"{col} = ?" for col in df.columns if col != 'id'])}
                WHERE id = ?
                """
                cursor = sqlite_conn.cursor()
                cursor.execute(update_query, tuple(row[col] for col in df.columns if col != 'id') + (row['id'],))
                sqlite_conn.commit()
                cursor.close()
        
        print("Backup concluído com sucesso.")
    
    except Exception as e:
        print(f"Erro ao fazer o backup: {e}")
    
    finally:
        engine.dispose()
        sqlite_conn.close()
        print("Conexão com o banco de dados encerrada.")
        
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

