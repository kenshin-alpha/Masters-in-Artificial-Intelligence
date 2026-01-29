"""
Feature Engineering Script for Crime Safety Classification Project
"""
import pandas as pd

# Load processed data
df = pd.read_csv('../data/processed_crime_data.csv')

# Example feature engineering: aggregate by location
# (Modify as needed based on actual columns)
def aggregate_by_location(df):
    # Group by location and calculate crime count and average severity
    if 'location' in df.columns and 'severity' in df.columns:
        agg = df.groupby('location').agg({
            'severity': ['mean', 'count']
        })
        agg.columns = ['avg_severity', 'crime_count']
        agg.reset_index(inplace=True)
        return agg
    else:
        print('Required columns not found.')
        return None

# Save engineered features
def save_features(agg):
    if agg is not None:
        agg.to_csv('../data/features_by_location.csv', index=False)
        print('Feature engineering complete. Features saved.')

if __name__ == '__main__':
    agg = aggregate_by_location(df)
    save_features(agg)
