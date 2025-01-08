import pandas as pd
from sqlalchemy import create_engine

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

