import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Feature distributions based on real educational research patterns
    # Attendance rate: mean 82%, std 14%, bounded between 35% and 100%
    attendance_rate = np.clip(np.random.normal(82, 14, n_samples), 35.0, 100.0)
    
    # Weekly study hours outside class: 1 to 24 hrs
    study_hours_weekly = np.clip(np.random.gamma(shape=3.0, scale=2.5, size=n_samples), 1.0, 24.0)
    
    # Past exam score: 30 to 98
    past_exam_score = np.clip(np.random.normal(68, 16, n_samples), 25.0, 99.0)
    
    # Commute distance (km): exponential distribution (many close, some very far)
    commute_distance_km = np.clip(np.random.exponential(scale=5.0, size=n_samples) + 0.5, 0.5, 25.0)
    
    # Internet access: 1 (Yes) or 0 (No) ~ 72% have internet
    internet_access = np.random.binomial(n=1, p=0.72, size=n_samples)
    
    # Parental involvement: 1 (Low), 2 (Medium), 3 (High)
    parental_involvement = np.random.choice([1, 2, 3], size=n_samples, p=[0.25, 0.50, 0.25])
    
    # Financial aid / fee support needed: 1 (Yes) or 0 (No) ~ 35%
    financial_aid_needed = np.random.binomial(n=1, p=0.35, size=n_samples)
    
    # Extracurricular participation: 1 (Yes) or 0 (No) ~ 45%
    extracurricular = np.random.binomial(n=1, p=0.45, size=n_samples)
    
    # 2. Risk Score Calculation (Log-odds approach for realistic probabilistic outcome)
    # Higher score = Higher risk of dropout / failure
    risk_latent = (
        - 0.08 * (attendance_rate - 75)
        - 0.18 * (study_hours_weekly - 8)
        - 0.06 * (past_exam_score - 60)
        + 0.12 * (commute_distance_km - 5)
        - 0.75 * internet_access
        - 0.65 * (parental_involvement - 2)
        + 0.85 * financial_aid_needed
        - 0.40 * extracurricular
        + np.random.normal(0, 0.8, n_samples) # Real-world unobserved variance
    )
    
    # Probability using sigmoid
    probability = 1.0 / (1.0 + np.exp(-risk_latent))
    
    # Target label: 1 = At-Risk, 0 = On-Track (threshold ~ 0.45 for conservative safety)
    at_risk = (probability > 0.45).astype(int)
    
    df = pd.DataFrame({
        'student_id': [f"STU-{1000 + i}" for i in range(n_samples)],
        'attendance_rate': np.round(attendance_rate, 1),
        'study_hours_weekly': np.round(study_hours_weekly, 1),
        'past_exam_score': np.round(past_exam_score, 1),
        'commute_distance_km': np.round(commute_distance_km, 1),
        'internet_access': internet_access,
        'parental_involvement': parental_involvement,
        'financial_aid_needed': financial_aid_needed,
        'extracurricular': extracurricular,
        'at_risk': at_risk
    })
    
    return df

def generate_sample_batch(n_samples=30, random_state=99):
    names = [
        "Aarav Khan", "Sara Ahmed", "Zain Ali", "Fatima Noor", "Bilal Tariq",
        "Ayesha Malik", "Usman Farooq", "Hania Raza", "Hamza Sheikh", "Maham Javed",
        "Omar Khalid", "Zoya Bashir", "Danish Rehman", "Anum Siddiqui", "Haris Mehmood",
        "Rida Qureshi", "Shahzaib Butt", "Maryam Nawaz", "Fahad Mustafa", "Laiba Shah",
        "Saad Aslam", "Sana Mir", "Moiz Baig", "Kiran Waqar", "Taha Hashmi",
        "Aiman Zia", "Nabeel Akhtar", "Hira Mani", "Waleed Safdar", "Nimra Chaudhry"
    ]
    
    df = generate_student_dataset(n_samples=n_samples, random_state=random_state)
    df.drop(columns=['at_risk'], inplace=True)
    df['student_name'] = names[:n_samples]
    # Reorder columns
    cols = ['student_id', 'student_name'] + [c for c in df.columns if c not in ['student_id', 'student_name']]
    return df[cols]

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dataset_path = os.path.join(current_dir, "students_dataset.csv")
    sample_batch_path = os.path.join(current_dir, "sample_batch_students.csv")
    
    print("Generating comprehensive student risk dataset...")
    df_main = generate_student_dataset(n_samples=2500)
    df_main.to_csv(main_dataset_path, index=False)
    print(f"Main dataset saved: {main_dataset_path} (Shape: {df_main.shape})")
    print(f"Risk distribution:\n{df_main['at_risk'].value_counts(normalize=True).round(3)}")
    
    print("\nGenerating sample batch test CSV for live app upload...")
    df_sample = generate_sample_batch(n_samples=30)
    df_sample.to_csv(sample_batch_path, index=False)
    print(f"Sample batch CSV saved: {sample_batch_path} (Shape: {df_sample.shape})")
