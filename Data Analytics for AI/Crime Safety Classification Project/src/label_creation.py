"""
Label Creation Script for Crime Safety Classification Project
"""
import pandas as pd

# Load features
df = pd.read_csv('../data/features_by_location.csv')

# Define safety levels based on crime count and average severity
# (Thresholds are examples; adjust as needed)
def assign_safety_level(row):
    if row['crime_count'] < 5 and row['avg_severity'] < 2:
        return 'safe'
    elif row['crime_count'] < 15 and row['avg_severity'] < 4:
        return 'moderately safe'
    else:
        return 'unsafe'

df['safety_level'] = df.apply(assign_safety_level, axis=1)

df.to_csv('../data/labeled_data.csv', index=False)
print('Label creation complete. Labeled data saved.')
