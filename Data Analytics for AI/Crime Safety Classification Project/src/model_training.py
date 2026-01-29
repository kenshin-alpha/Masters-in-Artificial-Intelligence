"""
Model Training Script for Crime Safety Classification Project
"""
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load labeled data
df = pd.read_csv('../data/labeled_data.csv')

# Prepare features and labels
X = df.drop(['location', 'safety_level'], axis=1)
y = df['safety_level']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=le.classes_))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Save model (optional)
import joblib
joblib.dump(clf, '../reports/random_forest_model.joblib')
print('Model training complete. Model saved.')
