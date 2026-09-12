# Logistics Delivery-Time Predictive Modeling

Predictive modeling and optimization pipeline for forecasting shipment
delivery time (hours) from route, load, traffic, and weather features,
built with pandas and scikit-learn.

## Structure
```
logistics-predictive-modeling/
├── data/
│   ├── logistics_delivery_dataset.csv   # simulated 600-shipment dataset
│   └── model_metrics.json               # generated evaluation results
├── src/
│   └── train_model.py                   # data prep, training, tuning, evaluation
├── docs/
│   └── Logistics_Predictive_Modeling_Report.docx
├── requirements.txt
└── README.md
```

## Problem
Predict `delivery_time_hours` for a shipment given: distance, weight,
number of stops, traffic conditions, weather, driver experience,
carrier, vehicle type, and region type (urban/suburban/rural).

## Models
- **Linear Regression** — interpretable baseline
- **Random Forest Regressor** — tuned via `GridSearchCV` (5-fold CV)
  over `n_estimators`, `max_depth`, `min_samples_leaf`

## Evaluation
RMSE, MAE, and R² on a held-out 20% test split, plus 5-fold
cross-validated RMSE for the tuned Random Forest. Feature importances
from the Random Forest are used to derive optimization
recommendations (route/distance planning, weather-aware scheduling,
stop-sequencing, traffic-aware dispatch).

## Usage
```bash
pip install -r requirements.txt
python src/train_model.py
```

## Report
See `docs/Logistics_Predictive_Modeling_Report.docx` for the full
methodology, results, and optimization recommendations.
