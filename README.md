
# Time Series Forecasting Project

## Overview

This project focuses on **time series analysis and forecasting** for weekly sales data of a retail department. It leverages classical statistical methods, machine learning models, and deep learning architectures to model and predict sales trends. The analysis spans from exploratory data visualization to the development and comparison of various forecasting models.

## Dataset

- **Source**: `project_dataset.csv`
- **Timeframe**: 2010 to 2012
- **Key Features**: `Weekly_Sales`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `IsHoliday`, etc.

## Project Stages

### 1. Data Exploration
- Filtered data for department ID 16.
- Identified trends and seasonal patterns.
- Visualized weekly sales and economic indicators.

### 2. Time Series Analysis
- Performed seasonal decomposition.
- Applied ACF and PACF for autocorrelation analysis.
- Conducted stationarity tests (ADF).

### 3. Classical Forecasting Models
- **Holt’s Linear Trend Model**
- **Holt-Winters Seasonal Model**
- **ARIMA (1,1,1)**
- **SARIMA (1,1,1)(1,1,1,52)**
- **SARIMAX** with external regressors

### 4. Feature Engineering
- Created lag features, rolling means, and time-based features.

### 5. Machine Learning Models
- **Random Forest**
- **XGBoost**
- **LightGBM**

### 6. Deep Learning Models
- **ANN** (Artificial Neural Network)
- **LSTM** (Long Short-Term Memory)
- **CNN** (Convolutional Neural Network)
- **RNN** (Recurrent Neural Network)
- **GRU** (Gated Recurrent Unit)

### 7. Prophet Forecasting
- Time-series regression using Facebook Prophet.

### 8. Model Evaluation
All models were evaluated using RMSE and MAE.
Results are saved to `model_evaluation_results.csv` and visualized in a grouped bar chart.

### 9. Deployment (Streamlit Dashboard)

An interactive sales prediction dashboard is deployed using **Streamlit**, offering:
- Manual input for economic and time-series features.
- On-demand predictions via FastAPI backend.
- Interactive plots comparing historical and predicted sales.
- Confidence intervals for prediction insights.
- Sales trends and performance metrics.

To run the dashboard:
```bash
streamlit run app.py
```
Ensure that the FastAPI server is running at `http://127.0.0.1:8000/predict`.

## Results Summary

The best-performing models (lowest RMSE) were:
- LightGBM (Machine Learning)
- GRU (Deep Learning)
- Prophet (Time Series Regression)

## Deployment Artifacts

- `random_forest_model.pkl`: Trained ML model for deployment.
- `app.py`: Streamlit dashboard interface.
- `model_evaluation_results.csv`: Performance summary.

---

> This project showcases a comprehensive comparison between classical, ML, DL, and real-time forecasting deployment using Streamlit.
