import pandas as pd

file_path = r'c:\Users\Kenshin\Documents\Data Analytics for AI\Crime_Safety_Classification_Project\data\RCD06.20251204131643.csv'
df = pd.read_csv(file_path)

print("Columns:", df.columns)
print("Unique Regions:", df['Garda Region'].unique())
print("Unique Years:", df['Year'].unique())
print("Unique Offence Types:", df['Type of Offence'].unique())
print("Unique Statistics:", df['Statistic Label'].unique())

# Check for more granular location data if available
# The file view showed 'Garda Region' but maybe there's another column?
# Based on the file view, it seems to be the only location column.
