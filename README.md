# Customer Churn Prediction

An end-to-end machine-learning project that predicts customer churn for a telecom company using the Telco Customer Churn dataset.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Customer churn — the loss of clients or subscribers — is one of the most critical metrics for any subscription-based business. This project builds a predictive model to identify customers who are likely to churn, enabling proactive retention strategies.

## Project Structure

```
customer-churn-prediction-ml/
│
├── data/
│   ├── raw/                        # Original, unprocessed data
│   └── processed/                  # Cleaned and transformed data
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py       # Data loading & cleaning
│   ├── feature_engineering.py      # Feature creation & selection
│   ├── model_training.py           # Model training & tuning
│   ├── evaluation.py               # Model evaluation & metrics
|
│
├── models/
│   └── trained_model.pkl           # Serialised trained model
│
├── app/
│   └── app.py                      # Streamlit web application
│
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The project uses the **Telco Customer Churn** dataset.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the EDA notebook
jupyter notebook notebooks/eda.ipynb

# Train the model
python src/model_training.py

# Evaluate the model
python src/evaluation.py
```

## Model Performance



## Streamlit App

```bash
streamlit run app/app.py
```

