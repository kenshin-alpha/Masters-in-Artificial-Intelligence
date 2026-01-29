import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed_crime_data.csv')

# Prepare X and y
# Drop non-feature columns
X = df.drop(columns=['Garda Region', 'Year', 'Total_Crime_Rate', 'Safety_Level'])
y = df['Safety_Level']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define CV
# Note: Small sample size (28). Min class size is 8. 
# Stratified 10-fold is impossible (requires min 10 samples per class).
# Using 4-fold CV instead.
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Define Models
models = {
    'Baseline': DummyClassifier(strategy='most_frequent'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

print("--- Model Performance (Accuracy) without Feature Selection ---")
results = {}
for name, model in models.items():
    # Pipeline: Scale -> Model
    # Scaling is important for NN and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n--- Feature Selection ---")

# Technique 1: SelectKBest (ANOVA F-value)
print("\n1. SelectKBest (k=5)")
selector_1 = SelectKBest(score_func=f_classif, k=5)
X_new_1 = selector_1.fit_transform(X, y_encoded)
selected_indices_1 = selector_1.get_support(indices=True)
print("Selected Features:", X.columns[selected_indices_1].tolist())

# Evaluate models with SelectKBest
print("Performance with SelectKBest:")
for name, model in models.items():
    if name == 'Baseline': continue
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=5)),
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")

# Technique 2: RFE (Recursive Feature Elimination) with Decision Tree
print("\n2. RFE (k=5) using Decision Tree")
rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=5)
rfe.fit(X, y_encoded)
selected_indices_2 = rfe.get_support(indices=True)
print("Selected Features:", X.columns[selected_indices_2].tolist())

# Evaluate models with RFE selected features (Pre-filtered X)
# Note: RFE is a wrapper, so we can't easily put it in a pipeline with other models 
# unless we use the same estimator for selection. 
# For simplicity, we'll just use the features selected by DT for all models here to compare.
X_rfe = X.iloc[:, selected_indices_2]

print("Performance with RFE Features:")
for name, model in models.items():
    if name == 'Baseline': continue
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X_rfe, y_encoded, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")

print("\n--- Conclusion ---")
print("Due to the small dataset size (28 samples), results may be unstable.")
print("Feature selection helps to reduce dimensionality and potentially improve generalization.")
