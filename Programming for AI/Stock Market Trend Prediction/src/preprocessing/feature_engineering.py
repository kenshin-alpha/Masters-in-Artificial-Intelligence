import pandas as pd
import numpy as np
import os

# Import load_data from clean_data module
from clean_data import load_data


PROCESSED_DIR = "C:\\Users\\Kenshin\\Documents\\Programming for AI\\Project CA2\\data\\processed"


def engineer_features(df):
    
    features_list = []
    
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Ensuring Close is numeric
        ticker_df['Close'] = pd.to_numeric(ticker_df['Close'], errors='coerce')
        ticker_df = ticker_df.dropna(subset=['Close'])

        # 1. First Feature - SMA 50
        ticker_df['SMA_50'] = ticker_df['Close'].rolling(window=50).mean()
        
        # 2. Trend (Ground Truth for classification)
        # Logic: Bullish if Close > SMA
        def classify_trend(row):
            if pd.isna(row['SMA_50']): 
                return "Neutral"
            return "Bullish" if row['Close'] > row['SMA_50'] else "Bearish"
            
        ticker_df['Trend'] = ticker_df.apply(classify_trend, axis=1)
        
        # Filtering Neutral for training
        ticker_df = ticker_df[ticker_df['Trend'] != 'Neutral'].copy()
        ticker_df['Target'] = (ticker_df['Trend'] == 'Bullish').astype(int)
        
        # 3. Price Change
        ticker_df['Price_Change'] = ticker_df['Close'].pct_change()
        
        # 4. Distance from SMA
        ticker_df['Distance_from_SMA'] = ((ticker_df['Close'] - ticker_df['SMA_50']) / ticker_df['SMA_50']) * 100
        
        # 5. Momentum 5d
        ticker_df['Momentum_5d'] = ticker_df['Close'].pct_change(periods=5)
        
        # 6. Volatility (5d std)
        ticker_df['Volatility'] = ticker_df['Close'].rolling(window=5).std()
        
        # 7. Target for Prediction: NEXT Day's Trend
        ticker_df['Next_Day_Target'] = ticker_df['Target'].shift(-1)
        
        features_list.append(ticker_df)
        
    full_df = pd.concat(features_list, ignore_index=True)
    full_df = full_df.dropna()  # Drop NaN from rolling/shift
    
    return full_df

  

df = load_data()

processed_df = engineer_features(df)
output_path = os.path.join(PROCESSED_DIR, "training_data.csv")
processed_df.to_csv(output_path, index=False)
print(f"Saved processed data to {output_path} ({len(processed_df)} rows)")
 
