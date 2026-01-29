import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure directories exist
os.makedirs('reports/figures', exist_ok=True)

# Load data
file_path = r'data/CJA07.20251204134405.csv'
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()
print("Available Columns:", df.columns.tolist())

# Filter for 'Recorded crime incidents'
if 'Statistic Label' in df.columns:
    df = df[df['Statistic Label'] == 'Recorded crime incidents'].copy()
else:
    print("Error: 'Statistic Label' column not found.")
    exit(1)

# Extract Division from Garda Station
# Format: "Station Name, Division Name"
# Some might not have a comma, so we handle that.
def extract_division(station_str):
    if ',' in station_str:
        return station_str.split(',')[-1].strip()
    return 'Unknown'

df['Division'] = df['Garda Station'].apply(extract_division)

# Pivot the data: Index=(Station, Division, Year), Columns=Type of Offence, Values=VALUE
df_pivot = df.pivot_table(
    index=['Garda Station', 'Division', 'Year'],
    columns='Type of Offence',
    values='VALUE',
    aggfunc='sum'
).reset_index()

# Fill missing values with 0 (assuming no crime recorded = 0)
df_pivot.fillna(0, inplace=True)

# Calculate Total Crime
offence_cols = df_pivot.columns[3:] # Skip Station, Division, Year
df_pivot['Total_Crime'] = df_pivot[offence_cols].sum(axis=1)

# Discretize Total Crime into 3 bins: Safe, Moderately Safe, Unsafe
# Using quantiles
df_pivot['Safety_Level'] = pd.qcut(df_pivot['Total_Crime'], q=3, labels=['Safe', 'Moderately Safe', 'Unsafe'])

print("Data Shape:", df_pivot.shape)
print("Safety Level Distribution:\n", df_pivot['Safety_Level'].value_counts())
print("Summary Statistics:\n", df_pivot['Total_Crime'].describe())

# Visual 1: Boxplot of Total Crime by Division (Top 10 Divisions by average crime)
top_divisions = df_pivot.groupby('Division')['Total_Crime'].mean().nlargest(10).index
df_top = df_pivot[df_pivot['Division'].isin(top_divisions)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='Division', y='Total_Crime', palette='viridis')
plt.title('Distribution of Total Crime by Division (Top 10)', fontsize=14)
plt.xlabel('Garda Division', fontsize=12)
plt.ylabel('Total Crime Incidents', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/visual_1_boxplot_new.png')
plt.close()

# Visual 2: Scatter Plot of two major crime types
# We need to find the exact column names from the new dataset
# Based on inspection: 'Theft and related offences (08)', 'Burglary and related offences (07)'
col_x = 'Theft and related offences (08)'
col_y = 'Burglary and related offences (07)'

if col_x in df_pivot.columns and col_y in df_pivot.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pivot, x=col_x, y=col_y, hue='Safety_Level', alpha=0.6, palette='coolwarm')
    plt.title(f'Safety Level: {col_x} vs {col_y}', fontsize=14)
    plt.xlabel(col_x, fontsize=12)
    plt.ylabel(col_y, fontsize=12)
    plt.legend(title='Safety Level')
    plt.tight_layout()
    plt.savefig('reports/figures/visual_2_scatter_new.png')
    plt.close()
else:
    print(f"Columns {col_x} or {col_y} not found. Available columns: {df_pivot.columns}")

# Save processed data
df_pivot.to_csv('data/processed_crime_data_new.csv', index=False)
print("Processed data saved to data/processed_crime_data_new.csv")
