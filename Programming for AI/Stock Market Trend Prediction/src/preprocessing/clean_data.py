import pandas as pd
import json
import os

# Configuration
RAW_DIR = "C:\\Users\\Kenshin\\Documents\\Programming for AI\\Project CA2\\data\\raw"

def load_data():

    data_frames = []
    
    for file in os.listdir(RAW_DIR):
        file_path = os.path.join(RAW_DIR, file)
        
        # Extract ticker from filename (e.g., stock_data_AAPL.csv -> AAPL)
        if file.startswith('stock_data_'):
            ticker = file.replace('stock_data_', '').replace('.csv', '').replace('.json', '')
        else:
            continue
        
        try:
            if file.endswith('.csv'):
                
                # Reading row 1 to get column names
                header_df = pd.read_csv(file_path, nrows=1)
                column_names = header_df.columns.tolist()
                
                # Reading the actual data starting from row 4 (index 3)
                df = pd.read_csv(file_path, skiprows=3, names=column_names)
                
                # Renaming 'Price' column to 'Date' (it contains the dates)
                if 'Price' in df.columns:
                    df.rename(columns={'Price': 'Date'}, inplace=True)
                    
            elif file.endswith('.json'):
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract data for each column
                rows = {}
                for key, values in data.items():
                    # Parse column name from tuple string like "('Close', 'AAPL')"
                    col_name = key.split("'")[1]  # Extract 'Close', 'High', etc.
                    
                    for timestamp, value in values.items():
                        timestamp_int = int(timestamp)
                        if timestamp_int not in rows:
                            rows[timestamp_int] = {}
                        rows[timestamp_int][col_name] = value
                
                # Converting to DataFrame
                df = pd.DataFrame.from_dict(rows, orient='index')
                df.index.name = 'Timestamp'
                df.reset_index(inplace=True)
                
                # Converting timestamp (ms) to date
                df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.drop('Timestamp', axis=1, inplace=True)
            
            else:
                continue
            
            # Ensuring required columns exist
            required = ['Date', 'Close']
            if not all(col in df.columns for col in required):
                print(f"Skipping {ticker}: Missing required columns. Found: {df.columns.tolist()}")
                continue
            
            # Converting Date to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Ensuring numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Ticker'] = ticker
            data_frames.append(df)
            print(f"Loaded {len(df)} rows for {ticker}")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")

    
    if not data_frames:
        raise ValueError("No valid data found in data/raw")
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} rows, {len(data_frames)} tickers")
    return combined_df


df = load_data()
print("\nData loaded successfully!")
print(df.info())
