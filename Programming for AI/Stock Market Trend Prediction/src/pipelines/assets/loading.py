import pandas as pd
import json
from dagster import asset, AssetExecutionContext, AssetCheckResult, asset_check
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from ..resources import DataStorageResource
from .transformation import training_dataset


@asset(
    description="Trained RandomForest model",
    group_name="loading",
    deps=[training_dataset],
)
def trained_model(
    context: AssetExecutionContext,
    storage: DataStorageResource,
    training_dataset: pd.DataFrame
):
    context.log.info("Training RandomForest model")
    
    # Preparing features and target
    feature_cols = ['SMA_50', 'Price_Change', 'Distance_from_SMA', 'Momentum_5d', 'Volatility']
    target_col = 'Next_Day_Target'
    
    X = training_dataset[feature_cols].copy()
    y = training_dataset[target_col].copy()
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    context.log.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Training the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    context.log.info(f"Model Performance - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
    context.log.info(f"Feature Importance:\n{feature_importance.to_string(index=False)}")
    
    # Saving the model
    model_path = storage.get_model_path("random_forest_model.pkl")
    joblib.dump(model, model_path)
    context.log.info(f"Model saved to {model_path}")
    
    return metrics


@asset(
    description="Model performance metrics",
    group_name="loading",
    deps=[trained_model],
)
def model_metrics(
    context: AssetExecutionContext,
    storage: DataStorageResource,
    trained_model: dict
):
    context.log.info("Saving model metrics")
    
    # Saving metrics
    metrics_path = storage.get_model_path("model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(trained_model, f, indent=2)
    
    context.log.info(f"Metrics saved to {metrics_path}")
    
    return trained_model


# Asset Check: Validating model performance
@asset_check(asset=trained_model)
def check_model_performance(trained_model: dict):
    checks_passed = True
    description = []
    
    # Minimum thresholds
    min_accuracy = 0.85
    min_precision = 0.85
    min_recall = 0.85
    
    accuracy = trained_model.get('accuracy', 0)
    precision = trained_model.get('precision', 0)
    recall = trained_model.get('recall', 0)
    
    # Check accuracy
    if accuracy >= min_accuracy:
        description.append(f"Accuracy: {accuracy:.4f} (>= {min_accuracy})")
    else:
        checks_passed = False
        description.append(f"ERROR: Accuracy: {accuracy:.4f} (< {min_accuracy})")
    
    # Check precision
    if precision >= min_precision:
        description.append(f"Precision: {precision:.4f} (>= {min_precision})")
    else:
        checks_passed = False
        description.append(f"ERROR: Precision: {precision:.4f} (< {min_precision})")
    
    # Check recall
    if recall >= min_recall:
        description.append(f"Recall: {recall:.4f} (>= {min_recall})")
    else:
        checks_passed = False
        description.append(f"ERROR: Recall: {recall:.4f} (< {min_recall})")
    
    return AssetCheckResult(
        passed=checks_passed,
        description="\n".join(description),
        metadata={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": trained_model.get('f1_score', 0),
        }
    )
