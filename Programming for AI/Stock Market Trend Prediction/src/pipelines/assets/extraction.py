import pandas as pd
import json
import os
import glob
from dagster import asset, AssetExecutionContext, AssetCheckResult, asset_check
from ..resources import DataStorageResource


@asset(
    description="Raw stock data loaded from CSV files",
    group_name="extraction",
)
def raw_stock_csv_data(context: AssetExecutionContext, storage: DataStorageResource):
    
    context.log.info(f"Loading CSV files from {storage.raw_dir}")
    
    data_frames = []
    csv_files = glob.glob(os.path.join(storage.raw_dir, "*.csv"))
    
    for file_path in csv_files:
        file = os.path.basename(file_path)
        
        if file.startswith('stock_data_'):
            ticker = file.replace('stock_data_', '').replace('.csv', '')
        else:
            continue
        
        try:
            header_df = pd.read_csv(file_path, nrows=1)
            column_names = header_df.columns.tolist()
            
            df = pd.read_csv(file_path, skiprows=3, names=column_names)
            
            # Renaming 'Price' column to 'Date'
            if 'Price' in df.columns:
                df.rename(columns={'Price': 'Date'}, inplace=True)
            
            # Converting Date to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Ensuring numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Ticker'] = ticker
            data_frames.append(df)
            context.log.info(f"Loaded {len(df)} rows for {ticker} from CSV")
            
        except Exception as e:
            context.log.error(f"Error loading {file}: {e}")
            continue
    
    if not data_frames:
        raise ValueError("No CSV files loaded")
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    context.log.info(f"Total CSV data: {len(combined_df)} rows from {len(data_frames)} tickers")
    
    return combined_df


@asset(
    description="Raw stock data loaded from JSON files",
    group_name="extraction",
)
def raw_stock_json_data(context: AssetExecutionContext, storage: DataStorageResource):
    context.log.info(f"Loading JSON files from {storage.raw_dir}")
    
    data_frames = []
    json_files = glob.glob(os.path.join(storage.raw_dir, "*.json"))
    
    for file_path in json_files:
        file = os.path.basename(file_path)
        
        if file.startswith('stock_data_'):
            ticker = file.replace('stock_data_', '').replace('.json', '')
        else:
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            rows = {}
            for key, values in data.items():
                col_name = key.split("'")[1]  # Extracting column name
                
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
            
            # Ensuring numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Ticker'] = ticker
            data_frames.append(df)
            context.log.info(f"Loaded {len(df)} rows for {ticker} from JSON")
            
        except Exception as e:
            context.log.error(f"Error loading {file}: {e}")
            continue
    
    if not data_frames:
        # Returning empty DataFrame if no JSON files
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    context.log.info(f"Total JSON data: {len(combined_df)} rows from {len(data_frames)} tickers")
    
    return combined_df


@asset(
    description="Combined raw stock data from both CSV and JSON sources",
    group_name="extraction",
    deps=[raw_stock_csv_data, raw_stock_json_data],
)
def combined_raw_data(
    context: AssetExecutionContext,
    raw_stock_csv_data: pd.DataFrame,
    raw_stock_json_data: pd.DataFrame
):
    context.log.info("Combining CSV and JSON data")
    
    # Combining data frames
    if len(raw_stock_json_data) > 0:
        combined = pd.concat([raw_stock_csv_data, raw_stock_json_data], ignore_index=True)
    else:
        combined = raw_stock_csv_data
    
    # Sorting by ticker and date
    combined = combined.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    tickers = combined['Ticker'].unique().tolist()
    context.log.info(f"Combined dataset: {len(combined)} rows from {len(tickers)} tickers: {tickers}")
    
    return combined


# Asset Check: Validate combined data quality
@asset_check(asset=combined_raw_data)
def check_combined_data_quality(combined_raw_data: pd.DataFrame):
   
    checks_passed = True
    description = []
    
    # Check 1: Data is not empty
    if len(combined_raw_data) == 0:
        checks_passed = False
        description.append("ERROR: Dataset is empty")
    else:
        description.append(f"Dataset contains {len(combined_raw_data)} rows")
    
    # Check 2: Required columns are present
    required_cols = ['Date', 'Close', 'Ticker']
    missing_cols = [col for col in required_cols if col not in combined_raw_data.columns]
    if missing_cols:
        checks_passed = False
        description.append(f"ERROR: Missing columns: {missing_cols}")
    else:
        description.append("All required columns present")
    
    # Check 3: No null values
    if checks_passed:
        null_dates = combined_raw_data['Date'].isnull().sum()
        null_close = combined_raw_data['Close'].isnull().sum()
        
        if null_dates > 0 or null_close > 0:
            checks_passed = False
            description.append(f"ERROR: Null values - Date: {null_dates}, Close: {null_close}")
        else:
            description.append("No null values in critical columns")
    
    # Check 4: Date range is reasonable
    if checks_passed:
        min_date = combined_raw_data['Date'].min()
        max_date = combined_raw_data['Date'].max()
        description.append(f"Date range: {min_date} to {max_date}")
    
    return AssetCheckResult(
        passed=checks_passed,
        description="\n".join(description),
        metadata={
            "row_count": len(combined_raw_data),
            "ticker_count": len(combined_raw_data['Ticker'].unique()) if checks_passed else 0,
        }
    )
