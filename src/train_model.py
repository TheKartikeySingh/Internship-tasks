"""
Logistics Delivery-Time Predictive Modeling Pipeline
=====================================================
Forecasts shipment delivery time (hours) from route, load, traffic,
weather, and carrier features, then extracts optimization insights
from the trained model.

Pipeline stages:
    1. Load & prepare data (impute missing values, encode categoricals)
    2. Train/test split
    3. Train baseline Linear Regression model
    4. Train Random Forest ensemble model
    5. Hyperparameter tuning via GridSearchCV + cross-validation
    6. Evaluate models (RMSE, MAE, R^2)
    7. Extract feature importance -> optimization recommendations
    8. Persist metrics + trained model artifacts

Run:
    python src/train_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/logistics_delivery_dataset.csv")
METRICS_PATH = Path("data/model_metrics.json")

TARGET = "delivery_time_hours"
NUMERIC_FEATURES = ["distance_km", "weight_kg", "num_stops", "traffic_index", "driver_experience_years"]
CATEGORICAL_FEATURES = ["carrier", "vehicle_type", "region_type", "weather"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"[Load] {len(df)} rows loaded")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Impute missing numerics with median, scale them, and one-hot encode categoricals."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"[{name}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()

    # --- Baseline: Linear Regression ---
    lr_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate("Linear Regression", lr_pipeline, X_test, y_test)

    # --- Ensemble: Random Forest (with hyperparameter tuning) ---
    rf_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 3],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"[Random Forest] Best params: {grid_search.best_params_}")
    rf_metrics = evaluate("Random Forest (tuned)", best_rf, X_test, y_test)

    # Cross-validated RMSE on full training data (robustness check)
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    cv_rmse_mean = float(-cv_scores.mean())
    cv_rmse_std = float(cv_scores.std())
    print(f"[Random Forest] 5-fold CV RMSE: {cv_rmse_mean:.3f} +/- {cv_rmse_std:.3f}")

    # --- Feature importance -> optimization insight ---
    feature_names = (
        NUMERIC_FEATURES
        + list(best_rf.named_steps["prep"].named_transformers_["cat"]
               .named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = best_rf.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n[Feature Importance] Top 8 drivers of delivery time:")
    print(importance_df.head(8).to_string(index=False))

    results = {
        "linear_regression": lr_metrics,
        "random_forest_tuned": rf_metrics,
        "random_forest_cv_rmse_mean": cv_rmse_mean,
        "random_forest_cv_rmse_std": cv_rmse_std,
        "best_params": grid_search.best_params_,
        "top_features": importance_df.head(8).to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[Done] Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
