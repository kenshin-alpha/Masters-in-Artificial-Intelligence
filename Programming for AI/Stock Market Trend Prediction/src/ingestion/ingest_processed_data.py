import yfinance as yf
import pandas as pd
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
import re

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
        csv_file = '../../data/processed/'

        files = glob.glob(os.path.join(csv_file, "*.csv"))
        print(f"Found {len(files)} files.")

        for csv_file in files:
            file_name = os.path.basename(csv_file)
            table_name = "training_data" 
            
            print(f"Processing '{file_name}' -> Table '{table_name}'")
            
            try:
                cursor.execute(f"""
                    DROP TABLE IF EXISTS {table_name};
                    CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    close_price NUMERIC(15, 6),
                    high_price NUMERIC(15, 6),
                    low_price NUMERIC(15, 6),
                    open_price NUMERIC(15, 6),
                    volume BIGINT,
                    trade_date DATE,
                    ticker VARCHAR(10),
                    sma_50 NUMERIC(15, 6),
                    trend VARCHAR(20),
                    target INTEGER,
                    price_change NUMERIC(15, 8),
                    distance_from_sma NUMERIC(15, 8),
                    momentum NUMERIC(15, 8),
                    volatility NUMERIC(15, 8),
                    next_day_target INTEGER
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
