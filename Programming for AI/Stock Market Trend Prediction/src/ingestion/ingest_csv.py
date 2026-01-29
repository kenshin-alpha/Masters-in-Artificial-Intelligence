import yfinance as yf
import pandas as pd
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
import re

def sanitize_table_name(filename):
    # Removing extension
    name = os.path.splitext(filename)[0]
    # Replacing non-alphanumeric characters with underscores and lowercase it
    clean_name = re.sub(r'[^a-zA-Z0-0]', '_', name).lower()
    # Ensuring it starts with a letter (standard SQL requirement)
    return f"raw_{clean_name}"

# Download AMZN data
stocks = 'AMZN'
data = yf.download(stocks, start='2015-01-01', end='2025-12-17')
print(data)
data.to_csv('../../data/raw/stock_data_AMZN.csv')


# Download NFLX data
stocks = 'NFLX'
data = yf.download(stocks, start='2015-01-01', end='2025-12-17')
print(data)
data.to_csv('../../data/raw/stock_data_NFLX.csv')


# Download GOOGL data
stocks = 'GOOGL'
data = yf.download(stocks, start='2015-01-01', end='2025-12-17')
print(data)
data.to_csv('../../data/raw/stock_data_GOOGL.csv')


# PostgreSQL Connection Test
dbname = "postgres"
user = "dap"
password = "dap"
host = "localhost"
port = "5432"
try:
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
        )
    cursor = conn.cursor()
    print(f"[Postgres] Connected to {dbname} on {host}:{port}")

    # --- 2. Load CSV Data ---
    try:
        csv_file = '../../data/raw/'

        files = glob.glob(os.path.join(csv_file, "*.csv"))
        print(f"Found {len(files)} files.")

        for csv_file in files:
            file_name = os.path.basename(csv_file)
            table_name = sanitize_table_name(file_name)
            
            print(f"Processing '{file_name}' -> Table '{table_name}'")
            
            try:
                cursor.execute(f"""
                    DROP TABLE IF EXISTS {table_name};
                    CREATE TABLE {table_name} (
                        column_1 TEXT,
                        column_2 TEXT,
                        column_3 TEXT,
                        column_4 TEXT,
                        column_5 TEXT,
                        column_6 TEXT
                    );
                """)

                # 2. Copy the data
                with open(csv_file, 'r', encoding='utf-8') as f:
                    sql = f"COPY {table_name} FROM STDIN WITH CSV"
                    cursor.copy_expert(sql, f)
                
                conn.commit()
                print(f"  Successfully created and populated {table_name}")

            except Exception as e:
                print(f"  Error processing {file_name}: {e}")
                conn.rollback()

    except Exception as e:
        print(f"Error: {e}")

except Exception as e:
    print(f"[Postgres] Connection failed: {e}")
finally:
    if conn:
        cursor.close()
