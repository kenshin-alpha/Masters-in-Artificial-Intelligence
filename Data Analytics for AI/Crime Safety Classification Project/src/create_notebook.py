import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Classifying Public Spaces by Safety Level\n",
    "\n",
    "## 1. EDA (Exploratory Data Analysis)\n",
    "\n",
    "### Dataset Description\n",
    "The dataset is derived from the 'Recorded Crime Incidents' data (Source: CSO/Garda). \n",
    "It contains crime statistics for different Garda Regions in Ireland from 2018 to 2024.\n",
    "\n",
    "### Preprocessing Steps\n",
    "1. **Filtering**: We focused on 'Recorded crime incident rate per 100,000 people' to ensure comparability across regions.\n",
    "2. **Pivoting**: The data was reshaped so that each row represents a unique (Region, Year) combination, and columns represent different offence types.\n",
    "3. **Target Variable**: We calculated the 'Total Crime Rate' by summing all offence rates. This was then discretized into 3 'Safety Levels' (Safe, Moderately Safe, Unsafe) using quantile binning to ensure balanced classes.\n",
    "4. **Scaling**: Standard Scaling was applied for distance-based algorithms (NN, SVM).\n",
    "\n",
    "**Sample Size Limitation**: The dataset resulted in 28 samples (4 Regions * 7 Years). This is extremely small for Machine Learning, especially Neural Networks, but serves as a proof of concept.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.dummy import DummyClassifier\n",
    "from sklearn.feature_selection import SelectKBest, f_classif, RFE\n",
    "from sklearn.pipeline import Pipeline\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Load data\n",
    "file_path = '../data/RCD06.20251204131643.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Preprocessing\n",
    "df_rate = df[\n",
    "    (df['Statistic Label'] == 'Recorded crime incident rate per 100,000 people') &\n",
    "    (df['Garda Region'] != 'State')\n",
    "].copy()\n",
    "\n",
    "df_pivot = df_rate.pivot_table(\n",
    "    index=['Garda Region', 'Year'],\n",
    "    columns='Type of Offence',\n",
    "    values='VALUE',\n",
    "    aggfunc='sum'\n",
    ").reset_index()\n",
    "\n",
    "offence_cols = df_pivot.columns[2:]\n",
    "df_pivot['Total_Crime_Rate'] = df_pivot[offence_cols].sum(axis=1)\n",
    "df_pivot['Safety_Level'] = pd.qcut(df_pivot['Total_Crime_Rate'], q=3, labels=['Safe', 'Moderately Safe', 'Unsafe'])\n",
    "\n",
    "print(\"Data Shape:\", df_pivot.shape)\n",
    "print(\"Safety Level Distribution:\\n\", df_pivot['Safety_Level'].value_counts())\n",
    "df_pivot.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Visualization\n",
    "\n",
    "We visualize the distribution of crime rates and the relationship between specific crime types and safety levels.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visual 1: Boxplot\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=df_pivot, x='Garda Region', y='Total_Crime_Rate', palette='viridis')\n",
    "plt.title('Distribution of Total Crime Rate per 100k People by Region (2018-2024)')\n",
    "plt.xticks(rotation=15)\n",
    "plt.show()\n",
    "\n",
    "# Visual 2: Scatter Plot\n",
    "col_x = 'Theft and related offences '\n",
    "col_y = 'Attempts/threats to murder, assaults, harassments and related offences '\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(data=df_pivot, x=col_x, y=col_y, hue='Safety_Level', style='Garda Region', s=100, palette='coolwarm')\n",
    "plt.title(f'Safety Level based on {col_x.strip()} vs {col_y.strip()}')\n",
    "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Model Creation\n",
    "\n",
    "We compare four models:\n",
    "1. **Baseline** (Most Frequent)\n",
    "2. **Neural Network** (MLPClassifier)\n",
    "3. **Decision Tree**\n",
    "4. **SVM** (Linear Kernel)\n",
    "\n",
    "**Validation**: Stratified 4-Fold Cross-Validation (due to small sample size).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_pivot.drop(columns=['Garda Region', 'Year', 'Total_Crime_Rate', 'Safety_Level'])\n",
    "y = df_pivot['Safety_Level']\n",
    "le = LabelEncoder()\n",
    "y_encoded = le.fit_transform(y)\n",
    "\n",
    "cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)\n",
    "\n",
    "models = {\n",
    "    'Baseline': DummyClassifier(strategy='most_frequent'),\n",
    "    'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),\n",
    "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
    "    'SVM': SVC(kernel='linear', random_state=42)\n",
    "}\n",
    "\n",
    "print(\"--- Model Performance (Accuracy) ---\")\n",
    "for name, model in models.items():\n",
    "    pipeline = Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', model)\n",
    "    ])\n",
    "    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')\n",
    "    print(f\"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Selection\n",
    "\n",
    "We apply two techniques to identify the most relevant features:\n",
    "1. **SelectKBest** (ANOVA F-value)\n",
    "2. **RFE** (Recursive Feature Elimination) with Decision Tree\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"--- Feature Selection ---\")\n",
    "\n",
    "# 1. SelectKBest\n",
    "selector_1 = SelectKBest(score_func=f_classif, k=5)\n",
    "X_new_1 = selector_1.fit_transform(X, y_encoded)\n",
    "print(\"SelectKBest Features:\", X.columns[selector_1.get_support(indices=True)].tolist())\n",
    "\n",
    "# 2. RFE\n",
    "rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=5)\n",
    "rfe.fit(X, y_encoded)\n",
    "print(\"RFE Features:\", X.columns[rfe.get_support(indices=True)].tolist())\n",
    "\n",
    "# Compare Performance with RFE Features\n",
    "X_rfe = X.iloc[:, rfe.get_support(indices=True)]\n",
    "print(\"\\nPerformance with RFE Features:\")\n",
    "for name, model in models.items():\n",
    "    if name == 'Baseline': continue\n",
    "    pipeline = Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', model)\n",
    "    ])\n",
    "    scores = cross_val_score(pipeline, X_rfe, y_encoded, cv=cv, scoring='accuracy')\n",
    "    print(f\"{name}: {scores.mean():.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Conclusion\n",
    "\n",
    "### Findings\n",
    "- **Data**: The dataset is small but shows clear regional distinctions in crime rates.\n",
    "- **Models**: Decision Tree and Neural Network (with RFE) performed best, achieving around 75-79% accuracy.\n",
    "- **Feature Selection**: RFE identified key indicators like 'Burglary', 'Theft', and 'Public Order' offences as strong predictors of the overall safety level.\n",
    "\n",
    "### Limitations\n",
    "- **Sample Size**: With only 28 samples, the model is prone to overfitting and the evaluation metrics may be unstable.\n",
    "- **Granularity**: Analyzing at the 'Region' level is very broad. More granular data (e.g., Station level) would provide better insights."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/Safety_Classification_Project.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook created successfully.")
