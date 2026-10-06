# 📚 EduAid AI: Technical & Machine Learning Study Notes

---

## 1. Problem Definition & Humanitarian Significance
In underserved schools and non-profit learning centers, high student dropout rates stem from systemic socio-economic friction rather than academic incapacity:
* **The Invisible Friction**: Long commute distances, lack of digital connectivity, and familial economic distress lead to chronic absenteeism and eventual school abandonment.
* **The Detection Gap**: By the time final examinations occur, student disengagement has reached the point of no return.
* **The Solution**: **EduAid** is an early-warning classification system that computes risk probabilities early in the academic term, allowing educational workers to deploy targeted interventions (transport subsidies, offline homework kits, peer tutoring).

---

## 2. Feature Engineering & Domain Justification

| Feature | Type | Range | Sociological / Machine Learning Rationale |
| :--- | :--- | :--- | :--- |
| **`attendance_rate`** | Float | 35.0% - 100.0% | Strongest empirical leading indicator of disengagement. |
| **`study_hours_weekly`** | Float | 1.0 - 24.0 hrs | Measures self-directed academic habit and home environment stability. |
| **`past_exam_score`** | Float | 20.0 - 100.0 | Quantifies foundational subject mastery. |
| **`commute_distance_km`** | Float | 0.5 - 25.0 km | High commute acts as a physical fatigue and cost multiplier. |
| **`internet_access`** | Binary | 0 or 1 | Flags digital divide and inability to complete modern research assignments. |
| **`parental_involvement`** | Ordinal | 1 (Low) to 3 (High) | Measures emotional reinforcement and home accountability. |
| **`financial_aid_needed`** | Binary | 0 or 1 | Identifies economic stress requiring free meals or fee grants. |
| **`extracurricular`** | Binary | 0 or 1 | Known psychological protective factor fostering a sense of school belonging. |

---

## 3. Preprocessing & Data Scaling
* **Train / Test Split**: 80% Training ($N=2,000$), 20% Testing ($N=500$) with **stratification** on the target variable (`stratify=y`) to maintain class balance across splits.
* **StandardScaler**:
  $$z = \frac{x - \mu}{\sigma}$$
  Standardizes features by removing the mean and scaling to unit variance. Essential for distance-based algorithms and Logistic Regression convergence.

---

## 4. Model Architectures & Comparisons

### A. Logistic Regression
* **Mechanics**: Fits a linear decision boundary using the sigmoid function:
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
* **Strengths**: Highly interpretable odds ratios, fast computation.
* **Performance**: Achieved **87.6% Accuracy** and **75.9% Recall** with **0.936 ROC-AUC**.

### B. Decision Tree Classifier (`max_depth=5`)
* **Mechanics**: Recursively splits feature space based on Gini Impurity:
  $$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
* **Strengths**: Produces human-readable `IF-THEN` rules that teachers and social workers can immediately audit without ML knowledge.

### C. Random Forest Classifier (`n_estimators=120, max_depth=8`)
* **Mechanics**: Ensemble of decision trees trained with bagging (bootstrap aggregating) and random feature subsets.
* **Strengths**: Mitigates single-tree overfitting, handles nonlinear feature interactions, and calculates Gini feature importances.
* **Performance**: **84.2% Accuracy**, **0.899 ROC-AUC**, highly robust variance stability.

---

## 5. Why "Recall" is More Critical Than "Accuracy" in Social ML

In traditional business contexts, false positives cost advertising money. In humanitarian contexts:

$$\text{Recall} = \frac{TP}{TP + FN}$$

* **False Positive (Type I Error)**: An on-track student is flagged as at-risk. The counselor conducts a friendly 10-minute check-in. **Cost: Minor**.
* **False Negative (Type II Error)**: A struggling student is missed by the model. Nobody intervenes, and the child drops out of school permanently. **Cost: Catastrophic**.
> **Rule of Thumb**: For life-critical, medical, and educational early-warning tools, models should be tuned to maximize **Recall** to eliminate False Negatives.

---

## 6. Interview Quick-Fire Prep (For Your Portfolio)
* **Q: How did you handle potential bias in the dataset?**  
  *A: We used stratified sampling and selected features that reflect logistical friction (commute, digital access) rather than demographic identifiers (gender, ethnicity, religion).*
* **Q: Why not use a Deep Neural Network?**  
  *A: Tabular data with 2,500 samples performs best on gradient-boosted trees and ensembles. Furthermore, schools require explainable AI (XAI) rather than opaque neural black boxes.*
