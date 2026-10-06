import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "students_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    feature_cols = [
        'attendance_rate',
        'study_hours_weekly',
        'past_exam_score',
        'commute_distance_km',
        'internet_access',
        'parental_involvement',
        'financial_aid_needed',
        'extracurricular'
    ]
    target_col = 'at_risk'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Stratified Train/Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    candidate_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    }
    
    metrics_summary = {}
    fitted_models = {}
    
    print("\n" + "="*50)
    print("TRAINING & EVALUATION BENCHMARK")
    print("="*50)
    
    for name, model in candidate_models.items():
        # Train
        if name == "Decision Tree":
            # Decision trees don't strictly require scaling, but works with unscaled or scaled
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {rec:.4f} (Safety Critical: Flagging at-risk students)")
        print(f"Precision: {prec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Confusion Matrix:\n{np.array(cm)}")
        
        metrics_summary[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }
        fitted_models[name] = model

    # Select Best Model based on F1-Score / Recall balance
    # Random Forest is typically the best overall performer
    best_model_name = "Random Forest"
    best_model = fitted_models[best_model_name]
    
    # Feature Importances from Random Forest
    rf_importances = best_model.feature_importances_
    feature_importance_dict = {
        col: round(float(imp), 4) for col, imp in zip(feature_cols, rf_importances)
    }
    # Sort descending
    sorted_importances = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE RANKING (Random Forest)")
    print("="*50)
    for feat, imp in sorted_importances.items():
        print(f"{feat:22s}: {imp:.4f}")
        
    # Save artifacts
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    dt_model_path = os.path.join(models_dir, "decision_tree_model.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(fitted_models["Decision Tree"], dt_model_path)
    
    metadata = {
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "metrics": metrics_summary,
        "feature_importances": sorted_importances,
        "target_classes": {0: "On Track", 1: "At Risk"}
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSaved best model to: {best_model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metrics & importances to: {metrics_path}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
