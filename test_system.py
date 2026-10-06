import os
import json
import joblib
import numpy as np
import pandas as pd

def run_system_tests():
    print("=" * 60)
    print("RUNNING EDUAID AI AUTOMATED SYSTEM TEST SUITE")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Test model files existence
    assert os.path.exists(os.path.join(models_dir, "best_model.pkl")), "best_model.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "scaler.pkl")), "scaler.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "decision_tree_model.pkl")), "decision_tree_model.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "model_metrics.json")), "model_metrics.json missing"
    print("[PASS] Test 1: All model artifact files exist and are verified.")
    
    # 2. Test loading artifacts
    best_model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    dt_model = joblib.load(os.path.join(models_dir, "decision_tree_model.pkl"))
    with open(os.path.join(models_dir, "model_metrics.json")) as f:
        metrics = json.load(f)
    print("[PASS] Test 2: Models and scalers loaded successfully.")
    
    # 3. Test High-Risk Student Prediction
    # (Low attendance, low study hours, low exam score, long commute, no internet, high financial aid needed)
    at_risk_student = np.array([[45.0, 2.0, 38.0, 15.0, 0, 1, 1, 0]])
    scaled_risk = scaler.transform(at_risk_student)
    pred_risk = best_model.predict(scaled_risk)[0]
    prob_risk = best_model.predict_proba(scaled_risk)[0][1]
    
    print(f"Risk Student Assessment -> Prediction: {pred_risk} (1=At-Risk), Probability: {prob_risk*100:.1f}%")
    assert pred_risk == 1, "Should correctly flag high-risk profile as at-risk!"
    print("[PASS] Test 3: High-Risk profile accurately classified.")
    
    # 4. Test High-Performance On-Track Student Prediction
    # (High attendance, high study hours, high score, short commute, internet, high parental involvement)
    safe_student = np.array([[96.0, 16.0, 88.0, 2.0, 1, 3, 0, 1]])
    scaled_safe = scaler.transform(safe_student)
    pred_safe = best_model.predict(scaled_safe)[0]
    prob_safe = best_model.predict_proba(scaled_safe)[0][1]
    
    print(f"Safe Student Assessment -> Prediction: {pred_safe} (0=On-Track), Probability: {prob_safe*100:.1f}%")
    assert pred_safe == 0, "Should correctly flag healthy profile as on-track!"
    print("[PASS] Test 4: On-Track student accurately classified.")
    
    # 5. Test Batch CSV loading
    sample_batch_file = os.path.join(data_dir, "sample_batch_students.csv")
    assert os.path.exists(sample_batch_file), "sample_batch_students.csv missing"
    df_batch = pd.read_csv(sample_batch_file)
    assert len(df_batch) == 30, "Expected 30 demo students"
    print(f"[PASS] Test 5: Batch dataset validated with {len(df_batch)} student profiles.")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! SYSTEM READY FOR DEPLOYMENT & PORTFOLIO.")
    print("=" * 60)

if __name__ == "__main__":
    run_system_tests()
