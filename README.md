# AQIFusionNet

**AQIFusionNet: A CNN-LSTM-GRU and XGBoost Ensemble for Short-Term AQI Forecasting in Delhi**

AQIFusionNet is a state-of-the-art air quality forecasting dashboard. It combines spatial feature extraction, sequential time-series gates, and tabular boosting trees into a robust ensemble framework to forecast future Air Quality Index (AQI) ratings and individual pollutants in Delhi NCR.

---

## 🛠️ Project Architecture

The architecture consists of a 4-stage modeling pipeline:
1. **CNN (Convolutional Neural Network)**: Extracts multi-variable spatial dependencies and chemical correlations.
2. **LSTM (Long Short-Term Memory)**: Captures long-term recurring seasonal trends (e.g. crop residual burning, winter inversions).
3. **GRU (Gated Recurrent Unit)**: Efficiently models short-term temporal changes and wind fluctuations.
4. **XGBoost (Extreme Gradient Boosting)**: Corrects predictions and handles non-linear regressions to yield the final forecast.

---

## 📂 Codebase Structure

```
AQIFusionNet/
├── app.py                     # Main Streamlit dashboard interface
├── predict.py                 # Seasonal prediction engine using historical datasets
├── model_loader.py            # Custom model & scaler loader
├── utils.py                   # ReportLab PDF generator and health advisory utilities
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation guide
├── trained_model.py           # End-to-end model training script
├── delhi_daily_cleaned.csv    # Cleaned historical daily AQI dataset
├── Delhi_day_ 2015-2020.xlsx  # Daily training dataset (reference)
├── Delhi_hour_2015-2020.xlsx # Hourly training dataset (reference)
├── assets/
│   └── styles.css             # Responsive layout stylesheet
└── models/
    ├── deep_learning_model.keras  # Production neural net weights
    ├── xgboost_model.json         # Production XGBoost refiner
    ├── scaler.joblib              # Features scaler
    ├── scaler_y.joblib            # Target unscaler
    └── ensemble_meta.json         # Ensemble metadata
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
Launch the Streamlit server directly:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to interact with the dashboard.

---

## 💎 Features Included
* **Futuristic Glassmorphic UI**: Adapts smoothly to both **Dark Cyberpunk** and **Light Glassmorphism** visual themes.
* **Delhi Seasonality Forecasts**: The prediction system loads data directly from `delhi_daily_cleaned.csv` to capture Delhi's true historical trends (monsoon drops vs. extreme winter peaks).
* **Interactive Plotly Visualizations**: Includes an AQI Dial Gauge, Pollutant Concentration Bar Chart, Pollutant Radar Distribution Fingerprint, a 30-Day Trend Horizon, and a Model Confidence Indicator.
* **Instant Export Reporting**: Download predictions as standard **CSV files** or **print-ready PDF reports** compiled with custom styling and health advisories.
