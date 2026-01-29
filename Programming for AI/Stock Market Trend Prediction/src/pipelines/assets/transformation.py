import pandas as pd
import numpy as np
from dagster import asset, AssetExecutionContext, AssetCheckResult, asset_check
from ..resources import DataStorageResource
from .extraction import combined_raw_data


@asset(
    description="Cleaned stock data with validated columns and types",
    group_name="transformation",
    deps=[combined_raw_data],
)
def cleaned_stock_data(
    context: AssetExecutionContext,
    combined_raw_data: pd.DataFrame
):
    context.log.info("Cleaning stock data")
    
    df = combined_raw_data.copy()
    
    # Ensuring required columns exist
    required = ['Date', 'Close', 'Ticker']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
    
    # Dropping rows with null dates or close prices
    initial_len = len(df)
    df = df.dropna(subset=['Date', 'Close'])
    dropped = initial_len - len(df)
    
    if dropped > 0:
        context.log.warning(f"Dropped {dropped} rows with null Date/Close values")
    
    # Sorting by ticker and date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    context.log.info(f"Cleaned data: {len(df)} rows from {len(df['Ticker'].unique())} tickers")
    
    return df


@asset(
    description="Engineered features including SMA, momentum, and volatility",
    group_name="transformation",
    deps=[cleaned_stock_data],
)
def engineered_features(
    context: AssetExecutionContext,
    cleaned_stock_data: pd.DataFrame
):
    context.log.info("Engineering features")
    
    features_list = []
    
    for ticker in cleaned_stock_data['Ticker'].unique():
        ticker_df = cleaned_stock_data[cleaned_stock_data['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Ensuring Close is numeric
        ticker_df['Close'] = pd.to_numeric(ticker_df['Close'], errors='coerce')
        ticker_df = ticker_df.dropna(subset=['Close'])
        
        # SMA 50
        ticker_df['SMA_50'] = ticker_df['Close'].rolling(window=50).mean()
        
        # Trend classification
        def classify_trend(row):
            if pd.isna(row['SMA_50']):
                return "Neutral"
            return "Bullish" if row['Close'] > row['SMA_50'] else "Bearish"
        
        ticker_df['Trend'] = ticker_df.apply(classify_trend, axis=1)
        
        # Filtering Neutral for training
        ticker_df = ticker_df[ticker_df['Trend'] != 'Neutral'].copy()
        ticker_df['Target'] = (ticker_df['Trend'] == 'Bullish').astype(int)
        
        # Price Change
        ticker_df['Price_Change'] = ticker_df['Close'].pct_change()
        
        # Distance from SMA
        ticker_df['Distance_from_SMA'] = ((ticker_df['Close'] - ticker_df['SMA_50']) / ticker_df['SMA_50']) * 100
        
        # Momentum 5d
        ticker_df['Momentum_5d'] = ticker_df['Close'].pct_change(periods=5)
        
        # Volatility (5d std)
        ticker_df['Volatility'] = ticker_df['Close'].rolling(window=5).std()
        
        # Next Day Target
        ticker_df['Next_Day_Target'] = ticker_df['Target'].shift(-1)
        
        features_list.append(ticker_df)
        context.log.info(f"Engineered features for {ticker}: {len(ticker_df)} rows")
    
    full_df = pd.concat(features_list, ignore_index=True)
    full_df = full_df.dropna()  # Drop NaN from rolling/shift
    
    context.log.info(f"Total engineered features: {len(full_df)} rows")
    
    return full_df


@asset(
    description="Final training dataset ready for model consumption",
    group_name="transformation",
    deps=[engineered_features],
)
def training_dataset(
    context: AssetExecutionContext,
    storage: DataStorageResource,
    engineered_features: pd.DataFrame
):

    context.log.info("Preparing training dataset")
    
    # Save to processed folder
    output_path = storage.get_processed_path("training_data.csv")
    engineered_features.to_csv(output_path, index=False)
    
    context.log.info(f"Saved training dataset: {output_path} ({len(engineered_features)} rows)")
    
    return engineered_features


# Asset Check: Validate engineered features
@asset_check(asset=engineered_features)
def check_feature_quality(engineered_features: pd.DataFrame):
    
    checks_passed = True
    description = []
    
    # Check 1: Required feature columns
    required_features = ['SMA_50', 'Price_Change', 'Distance_from_SMA', 'Momentum_5d', 'Volatility', 'Target']
    missing_features = [col for col in required_features if col not in engineered_features.columns]
    
    if missing_features:
        checks_passed = False
        description.append(f"ERROR: Missing features: {missing_features}")
    else:
        description.append("✓ All required features present")
    
    # Check 2: No infinite values
    if checks_passed:
        inf_counts = {}
        for col in required_features:
            if col in engineered_features.columns:
                inf_count = np.isinf(engineered_features[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
        
        if inf_counts:
            checks_passed = False
            description.append(f"ERROR: Infinite values found: {inf_counts}")
        else:
            description.append("✓ No infinite values")
    
    # Check 3: Class balance
    if checks_passed and 'Target' in engineered_features.columns:
        target_counts = engineered_features['Target'].value_counts()
        bullish_pct = (target_counts.get(1, 0) / len(engineered_features)) * 100
        bearish_pct = (target_counts.get(0, 0) / len(engineered_features)) * 100
        
        description.append(f"✓ Class balance - Bullish: {bullish_pct:.1f}%, Bearish: {bearish_pct:.1f}%")
        
        # Warn if severe imbalance
        if bullish_pct < 20 or bullish_pct > 80:
            description.append(f"WARNING: Severe class imbalance detected")
    
    return AssetCheckResult(
        passed=checks_passed,
        description="\n".join(description),
        metadata={
            "row_count": len(engineered_features),
            "feature_count": len([col for col in required_features if col in engineered_features.columns]),
        }
    )
