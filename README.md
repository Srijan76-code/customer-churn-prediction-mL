<h1 align="center">Telecom Customer Churn Prediction System</h1>

<p align="center">
  <img src="PLACEHOLDER_FOR_HERO_IMAGE_OR_SCREENSHOT" alt="Project Banner" width="100%">
</p>

<p align="center">
    <a href="PLACEHOLDER_FOR_HOSTED_LINK" target="_blank">
        <img src="https://img.shields.io/badge/Live_Demo-Hosted_App-blue?style=for-the-badge&logo=appveyor" alt="Live Demo">
    </a>
    <br/>
    <em>Predicting and mitigating customer churn in the telecommunications sector using optimized machine learning models.</em>
</p>

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Data Description](#-data-description)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Feature Engineering & Preprocessing](#-feature-engineering--preprocessing)
- [Methodology](#-methodology)
- [Model Evaluation & Optimization](#-model-evaluation--optimization)
- [Final Model Selected](#-final-model-selected)
- [Application Screenshots](#-application-screenshots)
- [Installation & Usage](#-installation--usage)

---

## Problem Statement

Customer churn represents the event in which a subscriber terminates their relationship with a telecom provider. In subscription-based businesses, churn directly impacts:

- **Revenue stability**
- **Customer Lifetime Value (CLV)**
- **Acquisition cost efficiency**

**Objective:** To design and deploy a predictive machine learning system capable of identifying customers who are at high risk of churn before the event occurs.

We model churn as a supervised binary classification problem:

- `y = 1` if the customer churns
- `y = 0` if the customer stays

From a business cost perspective:
$$Cost_{FalseNegative} > Cost_{FalsePositive}$$

_Losing a customer is significantly more expensive than offering a retention incentive to a loyal one._ Therefore, the system is explicitly optimized to prioritize **Recall and ROC-AUC** over naive accuracy, ensuring we capture as many true churners as possible.

---

## Data Description

### Dataset Overview

- **Source:** Telco Customer Churn
- **Total Observations:** 7,043
- **Raw Features:** 38 columns

The dataset spans four major behavioral pillars:

1. **Demographics:** Age, gender, dependents.
2. **Tenure & Account History:** How long they have been with the company.
3. **Service Penetration:** Internet, phone, multiple lines, streaming, security.
4. **Financial & Billing Behavior:** Monthly charges, total charges, payment methods, contract type.

### Target Engineering

The original target column `Customer Status` contained three categories: _Stayed, Churned, Joined_.
We converted this into a binary target:

```python
Churn = 1 if Customer Status == "Churned" else 0
```

The original `Customer Status`, along with leakage-prone columns (`Churn Category`, `Churn Reason`), was removed.

---

## Exploratory Data Analysis (EDA)

### 1. Class Imbalance

- **Stayed:** ~73.5%
- **Churned:** ~26.5%

<p align="center">
  <img src="images/class_imbalance.png" alt="Class Imbalance Distribution" width="60%">
</p>

Because of this imbalance, a naive model that predicts "Stayed" every time would achieve ~73% accuracy. Thus, we evaluate based on **ROC-AUC, Recall, and F1-score**.

### 2. Target Leakage Prevention

Dropped Columns: `Customer ID`, `Customer Status` (after encoding), `Churn Category`, `Churn Reason`.
_Reason:_ Using post-churn survey data essentially gives the model the "answer key," preventing real-world generalization.

### 3. Geographical Data Removal

Dropped Columns: `City`, `Zip Code`, `Latitude`, `Longitude`.
_Reason:_ High cardinality categorical features create dimensional explosion without providing sufficient generalized signals.

### 4. Structural Missing Value Treatment

Instead of deleting rows, contextual imputation was applied preserving all 7,043 records:

- **Phone Service Logic:** If `Phone Service == "No"`, then `Avg Monthly Long Distance Charges = 0`.
- **Internet Service Logic:** If `Internet Service == "No"`, then related features (Online Security, Backup, Device Protection, Tech Support, Streaming, Unlimited Data) were set to `"No Internet Service"`.

### 5. Outlier Analysis

Outliers were detected via the IQR method. Outliers in `Avg Monthly GB Download` and `Avg Monthly Long Distance Charges` were **retained** because heavy usage is legitimate customer behavior containing vital churn signals.

---

## Feature Engineering & Preprocessing

### 1. Engineered Features

- **`Revenue_Per_Month`**: Derived from `Total Revenue / Tenure in Months`. This captures total monetization (base subscription + overages, refunds, long distance) representing a stronger signal of actual customer value intensity than standard monthly charges.
- **`Is_Monthly_Contract`**: A binary flag (1 if `Contract == "Month-to-Month"`, else 0). Month-to-month customers exhibit significantly higher historic churn rates.

### 2. Encoding Strategies

- **Binary Encoding:** Mapped columns like `Gender`, `Married`, `Phone Service`, `Paperless Billing`, and `Internet Service` to 0/1 to align with Logistic Regression assumptions.
- **One-Hot Encoding:** Applied to multi-class variables (`Offer`, `Multiple Lines`, `Internet Type`, `Payment Method`, etc.) using `drop_first=True` to eliminate multi-collinearity (the dummy variable trap).

### 3. Feature Scaling

Applied `StandardScaler` to ensure balanced contributions across features during Gradient Descent optimization. The scaler was fit exclusively on the training set to prevent data leakage.

---

## Methodology

### Data Split

- **Training Set:** 80% | **Testing Set:** 20%
- **Stratification:** Actively preserved class distribution proportions across splits.
- **Random Seed:** 42

### Models Evaluated

1. **Logistic Regression:** Trained with default settings but evaluated under different probability thresholds to align with the business cost sensitivity (reducing False Negatives).
2. **Decision Tree:** Optimized purely through structural depth tuning (`max_depth`) to balance bias and variance.

---

## Model Evaluation & Optimization

### Logistic Regression — Threshold Analysis

By lowering the decision threshold from 0.5 to 0.3, we traded a small drop in overall accuracy for a massive gain in Recall (finding more churners).

| Threshold          | Accuracy   | Churn Recall | Churn F1 |
| ------------------ | ---------- | ------------ | -------- |
| 0.5 (Baseline)     | 0.8200     | 0.68         | 0.64     |
| 0.4                | 0.8300     | 0.75         | 0.70     |
| **0.3 (Selected)** | **0.8077** | **0.83**     | **0.70** |

**Confusion Matrix Summary (Threshold = 0.3):**

- **True Negatives:** 828 | **False Positives:** 208
- **False Negatives:** 63 | **True Positives:** 310
  _(Only 63 churners missed!)_

<p align="center">
  <img src="images/cm_logistic_regression.png" alt="Confusion Matrix - Logistic Regression" width="60%">
</p>

### Decision Tree — Depth Tuning

Evaluated depths from 2 to 20, finding the optimal balance at **Depth 14**.

- **Depth 2:** Recall 0.23 (Severe Underfitting)
- **Depth 14:** Train Acc 0.889 | Test Acc 0.833 | Recall 0.61 | F1 0.66
- Depths beyond 15 plateaued, indicating model capacity saturation.

<p align="center">
  <img src="images/cm_decision_tree.png" alt="Confusion Matrix - Decision Tree" width="60%">
</p>

---

## Final Model Selected

| Model                    | Accuracy   | Churn Recall | ROC-AUC    | Strength                 |
| ------------------------ | ---------- | ------------ | ---------- | ------------------------ |
| Logistic (0.5)           | 0.8200     | 0.68         | 0.8876     | Balanced                 |
| Decision Tree (Depth 14) | 0.8330     | 0.61         | 0.8451     | Non-linear modeling      |
| **Logistic (0.3)**       | **0.8077** | **0.83**     | **0.8876** | **Best churn detection** |

**Winning Model: Logistic Regression (Threshold = 0.3)**

**Why this model?**

- **Highest ROC-AUC (0.8876)**
- **Strongest Churn Recall (0.83)**
- **Lowest False Negatives (63 missed vs higher in others)**
- Highly interpretable, stable generalization, and perfectly aligned with the cost-sensitive business objective.

---

## Application Screenshots

|                                        Dashboard                                         |                                     Prediction Results                                     |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| <img src="PLACEHOLDER_FOR_DASHBOARD_SCREENSHOT" width="100%" alt="Dashboard Screenshot"> | <img src="PLACEHOLDER_FOR_PREDICTION_SCREENSHOT" width="100%" alt="Prediction Screenshot"> |
|                           _System Overview & Upload Interface_                           |                         _Batch Prediction & Probabilities output_                          |

## Installation & Usage

_(Assuming you have Python 3.8+ installed)_

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction-mL
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application:**

```bash
streamlit run app/app.py
```

_This will launch the web application where you can upload customer data and get churn predictions._

---
