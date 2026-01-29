import json
import os
import yfinance as yf
import pandas as pd
from pymongo import MongoClient

# Download AAPL data
stocks = 'AAPL'
data = yf.download(stocks, start='2015-01-01', end='2025-12-17')
print(data)
# Save to JSON
data.to_json('../../data/raw/stock_data_AAPL.json')


SOURCE_DIR = "C:/Users/Kenshin/Documents/Programming for AI/Project CA2/data/raw"
MONGO_URI = "mongodb://127.0.0.1:27017/"


client = MongoClient(MONGO_URI)
    
    
db_path = SOURCE_DIR
        
print(f"\nChecking directory: {db_path}")
if os.path.isdir(db_path):
    print(f"--- Processing Database: raw_stock_data_aapl ---")
    db = client["raw_stock_data_aapl"]
            
for file_name in os.listdir(db_path):
    if file_name.endswith(".json"):
        collection_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(db_path, file_name)

        print(f"Importing {file_name} into raw_stock_data_aapl.{collection_name}...")
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                            
                # Ensure data is a list for insert_many
                if not isinstance(data, list):
                    data = [data]
                            
                # Replicate --drop: Delete existing data first
                coll = db[collection_name]
                coll.drop() 
                            
                # Insert the data
                if data:
                    coll.insert_many(data)
                    print(f"Successfully imported {len(data)} documents.")
                                
            except Exception as e:
                print(f"Failed to import {file_name}: {e}")

client.close()
print("\nAll imports completed.")
