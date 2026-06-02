"""
Traffic Demand Prediction Pipeline
===================================
Gridlock Hackathon 2.0 — Flipkart
Author: Sujeet Jha | NSUT, B.Tech CSE (Big Data Analytics)

End-to-end ML pipeline for urban traffic demand forecasting.
Dataset: 77K+ training samples with temporal and spatial features.

Pipeline stages:
    1. Data Loading & Exploration
    2. Preprocessing & Feature Engineering
    3. Model Training (Random Forest + XGBoost comparison)
    4. Hyperparameter Tuning (RandomizedSearchCV)
    5. Evaluation & Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data/traffic_data.csv"   # Replace with actual dataset path
MODEL_DIR   = "models"
PLOTS_DIR   = "plots"
TARGET_COL  = "demand"
N_SAMPLES   = 77000   # synthetic demo if no real data

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────
def load_or_generate_data(path: str, n: int = N_SAMPLES) -> pd.DataFrame:
    """Load CSV or generate synthetic traffic data for demonstration."""
    if os.path.exists(path):
        print(f"[INFO] Loading data from {path}")
        df = pd.read_csv(path)
    else:
        print(f"[INFO] Dataset not found. Generating {n} synthetic samples...")
        hours      = np.random.randint(0, 24, n)
        day_of_week = np.random.randint(0, 7, n)
        month      = np.random.randint(1, 13, n)
        zone       = np.random.choice(["Zone_A", "Zone_B", "Zone_C", "Zone_D"], n)
        weather    = np.random.choice(["clear", "rain", "fog", "snow"], n,
                                       p=[0.6, 0.2, 0.1, 0.1])
        is_holiday = np.random.choice([0, 1], n, p=[0.93, 0.07])
        temp_c     = np.random.normal(22, 8, n)

        # Demand: realistic signal with noise
        base    = 100 + 80 * np.sin(np.pi * hours / 12)
        wk_pen  = np.where(day_of_week >= 5, -20, 10)
        wx_pen  = np.select(
            [weather == "rain", weather == "fog", weather == "snow"],
            [-15, -25, -40], default=0)
        hol_pen = np.where(is_holiday == 1, -30, 0)
        demand  = (base + wk_pen + wx_pen + hol_pen
                   + np.random.normal(0, 12, n)).clip(0)

        df = pd.DataFrame({
            "hour": hours, "day_of_week": day_of_week, "month": month,
            "zone": zone, "weather": weather, "is_holiday": is_holiday,
            "temperature_c": temp_c.round(1), TARGET_COL: demand.round(1)
        })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[INFO] Synthetic data saved to {path}")
    return df


# ─────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def exploratory_analysis(df: pd.DataFrame) -> None:
    print("\n" + "═"*55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("═"*55)
    print(f"  Shape        : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")
    print(f"  Target mean  : {df[TARGET_COL].mean():.2f}")
    print(f"  Target std   : {df[TARGET_COL].std():.2f}")
    print("═"*55 + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Traffic Demand — EDA", fontsize=14, fontweight="bold")

    # Demand distribution
    axes[0].hist(df[TARGET_COL], bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Demand"); axes[0].set_ylabel("Frequency")

    # Hourly average demand
    hourly = df.groupby("hour")[TARGET_COL].mean()
    axes[1].plot(hourly.index, hourly.values, color="#16a34a", linewidth=2.5, marker="o", markersize=4)
    axes[1].set_title("Avg Demand by Hour")
    axes[1].set_xlabel("Hour of Day"); axes[1].set_ylabel("Avg Demand")
    axes[1].fill_between(hourly.index, hourly.values, alpha=0.15, color="#16a34a")

    # Demand by weather
    df.groupby("weather")[TARGET_COL].mean().sort_values().plot(
        kind="barh", ax=axes[2], color=["#ef4444","#f97316","#3b82f6","#22c55e"])
    axes[2].set_title("Avg Demand by Weather")
    axes[2].set_xlabel("Avg Demand")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] EDA plot saved → {PLOTS_DIR}/eda.png")


# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Feature engineering:
      - Cyclical encoding for hour (sin/cos) to preserve periodicity
      - Cyclical encoding for day_of_week and month
      - Label encoding for categorical columns
      - Peak-hour binary flag
    """
    df = df.copy()

    # Cyclical time features
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Peak-hour flag (7–9 AM and 5–8 PM)
    df["is_peak"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 20) else 0)

    # Categorical encoding
    le_zone    = LabelEncoder()
    le_weather = LabelEncoder()
    df["zone_enc"]    = le_zone.fit_transform(df["zone"])
    df["weather_enc"] = le_weather.fit_transform(df["weather"])

    FEATURE_COLS = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_peak", "is_holiday",
        "temperature_c", "zone_enc", "weather_enc"
    ]
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"[INFO] Features used  : {FEATURE_COLS}")
    print(f"[INFO] Feature matrix : {X.shape}")
    return X, y, FEATURE_COLS, le_zone, le_weather


# ─────────────────────────────────────────────
# STEP 4: MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    """Train Random Forest baseline and tuned variant."""

    print("\n[INFO] Training Random Forest baseline...")
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)

    print("[INFO] Hyperparameter tuning with RandomizedSearchCV (this may take ~1 min)...")
    param_dist = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2", 0.5],
    }
    rf_tuned = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=20, cv=5, scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_tuned.fit(X_train, y_train)

    print(f"[INFO] Best params: {rf_tuned.best_params_}")
    return rf_baseline, rf_tuned.best_estimator_


# ─────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, label="Model") -> dict:
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    print(f"\n  ── {label} ──")
    print(f"     MAE  : {mae:.3f}")
    print(f"     RMSE : {rmse:.3f}")
    print(f"     R²   : {r2:.4f}")
    print(f"     MAPE : {mape:.2f}%")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2,
            "MAPE": mape, "y_pred": y_pred}


def plot_results(results: list, y_test, feature_names, best_model) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Traffic Demand Prediction — Results", fontsize=14, fontweight="bold")

    colors = ["#3b82f6", "#16a34a", "#ef4444"]
    for i, res in enumerate(results):
        axes[0].scatter(y_test[:500], res["y_pred"][:500],
                        alpha=0.4, s=10, color=colors[i], label=res["label"])
    lims = [y_test.min(), y_test.max()]
    axes[0].plot(lims, lims, "k--", linewidth=1.5, label="Perfect")
    axes[0].set_title("Predicted vs Actual"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Actual Demand"); axes[0].set_ylabel("Predicted Demand")

    # Residuals for best model
    best_res = results[-1]
    residuals = y_test.values - best_res["y_pred"]
    axes[1].hist(residuals, bins=50, color="#8b5cf6", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residuals — {best_res['label']}")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances.sort_values().tail(8).plot(
        kind="barh", ax=axes[2], color="#f59e0b")
    axes[2].set_title("Top Feature Importances")
    axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Results plot saved → {PLOTS_DIR}/results.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  TRAFFIC DEMAND PREDICTION PIPELINE")
    print("  Gridlock Hackathon 2.0 — Flipkart | Sujeet Jha")
    print("█"*55)

    # 1. Load data
    df = load_or_generate_data(DATA_PATH)

    # 2. EDA
    exploratory_analysis(df)

    # 3. Preprocess
    X, y, feature_names, le_zone, le_weather = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"\n[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 4. Train
    rf_base, rf_tuned = train_models(X_train, y_train)

    # 5. Evaluate
    print("\n" + "═"*55)
    print("  EVALUATION RESULTS")
    print("═"*55)
    res_base  = evaluate(rf_base,  X_test, y_test, "RF Baseline")
    res_tuned = evaluate(rf_tuned, X_test, y_test, "RF Tuned")

    # 6. Cross-validation score
    cv_scores = cross_val_score(rf_tuned, X, y, cv=5,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"\n  5-Fold CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 7. Plots
    plot_results([res_base, res_tuned], y_test, feature_names, rf_tuned)

    # 8. Save model
    model_path = f"{MODEL_DIR}/rf_tuned_traffic.pkl"
    joblib.dump({"model": rf_tuned, "features": feature_names,
                 "le_zone": le_zone, "le_weather": le_weather}, model_path)
    print(f"\n[INFO] Model saved → {model_path}")

    print("\n" + "█"*55)
    print("  PIPELINE COMPLETE")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()