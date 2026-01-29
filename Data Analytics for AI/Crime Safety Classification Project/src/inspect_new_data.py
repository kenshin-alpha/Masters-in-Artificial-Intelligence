import pandas as pd

file_path = r'c:\Users\Kenshin\Documents\Data Analytics for AI\Crime_Safety_Classification_Project\data\CJA07.20251204134405.csv'
df = pd.read_csv(file_path)

print("Columns:", df.columns)
print("First 5 rows:\n", df.head())
print("Unique Stations:", df['Garda Station'].nunique())
print("Unique Years:", df['Year'].unique())
print("Unique Offence Types:", df['Type of Offence'].unique())
