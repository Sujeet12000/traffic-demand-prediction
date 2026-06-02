"""
Gridlock Hackathon 2.0 — Strategy v2: Exact Lookup + Target Encoding + RF
==========================================================================

THEORY:
  Top teams (score=100) exploit the fact that traffic demand at a given
  (geohash, day, timestamp) is nearly deterministic from history.
  If test rows share these 3 keys with training rows, the historical
  mean demand is a near-perfect predictor.

CHANGES FROM BASELINE (main_86_best.py):
  1. exact_demand feature: mean demand for (geohash, day, timestamp) in train
     → also used as DIRECT PREDICTION when an exact match exists in test
  2. geo_mean:     mean demand per geohash (location baseline)
  3. geo_ts_mean:  mean demand per (geohash, timestamp) (time-of-day at location)
  4. geo_day_mean: mean demand per (geohash, day) (day pattern at location)
  5. RF params UNCHANGED: n_estimators=100, random_state=42

DOES NOT OVERWRITE: main_86_best.py, submission_86_best.csv
OUTPUT:            submission_v2_lookup.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── 1. Load ────────────────────────────────────────────────────────────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# ── 2. Impute (identical to baseline) ─────────────────────────────────────────
for col in ["RoadType", "Weather"]:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col]  = test[col].fillna(mode_val)

train_temp_mean = train["Temperature"].mean()
train["Temperature"] = train["Temperature"].fillna(train_temp_mean)
test["Temperature"]  = test["Temperature"].fillna(train_temp_mean)  # use TRAIN mean for test

global_mean = train["demand"].mean()

# ── 3. Exact-Match Lookup: (geohash, day, timestamp) → historical mean ─────────
#    Core idea: test rows that share these 3 keys with train have a near-perfect
#    predictor available — the historical average demand for that slot.
LOOKUP_KEYS = ["geohash", "day", "timestamp"]

exact_agg = (
    train.groupby(LOOKUP_KEYS)["demand"]
         .mean()
         .reset_index()
         .rename(columns={"demand": "exact_demand"})
)

train = train.merge(exact_agg, on=LOOKUP_KEYS, how="left")
test  = test.merge(exact_agg, on=LOOKUP_KEYS, how="left")

# Save raw lookup values (NaN = no train match) BEFORE we fill for feature use
test_raw_lookup = test["exact_demand"].copy()
n_match = test_raw_lookup.notna().sum()
print(f"[INFO] Test rows with exact (geohash+day+timestamp) match: "
      f"{n_match}/{len(test)} ({100*n_match/len(test):.1f}%)")
# If this is >80%, v3 (pure lookup) will likely score very high too.

# Fill NaN so exact_demand is usable as a model feature
train["exact_demand"] = train["exact_demand"].fillna(global_mean)
test["exact_demand"]  = test["exact_demand"].fillna(global_mean)

# ── 4. Additional Target-Encoding Features ─────────────────────────────────────
#    Give RF real demand signals instead of arbitrary integer label IDs.
def add_te(tr, te, keys, col_name):
    agg = tr.groupby(keys)["demand"].mean().reset_index()
    agg.columns = keys + [col_name]
    tr = tr.merge(agg, on=keys, how="left")
    te = te.merge(agg, on=keys, how="left")
    tr[col_name] = tr[col_name].fillna(global_mean)
    te[col_name] = te[col_name].fillna(global_mean)
    return tr, te

# Coarser → finer grained (each tells RF something different)
train, test = add_te(train, test, ["geohash"],               "geo_mean")      # location baseline
train, test = add_te(train, test, ["geohash", "timestamp"],  "geo_ts_mean")   # time-of-day at location
train, test = add_te(train, test, ["geohash", "day"],        "geo_day_mean")  # day-of-week at location

# ── 5. Label-Encode Categoricals (identical to baseline) ──────────────────────
cat_cols = ["geohash", "timestamp", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]])
    le.fit(combined.astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── 6. Train RF (same params as baseline) ─────────────────────────────────────
X = train.drop("demand", axis=1)
y = train["demand"]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Feature importance diagnostic — if exact_demand & geo_ts_mean rank high,
# target encoding is working as expected.
fi = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
print("\n[INFO] Top 10 Feature Importances:")
print(fi.to_string())

# ── 7. Predict with Lookup Override ───────────────────────────────────────────
X_test   = test[X.columns]
rf_preds = model.predict(X_test)

# Key: where we have an exact historical match, bypass RF entirely.
# RF is only used as a fallback for unseen (geohash, day, timestamp) combos.
final_preds = np.where(
    test_raw_lookup.notna(),   # exact match found → use historical mean
    test_raw_lookup,
    rf_preds                   # no match → RF fallback
)

print(f"\n[INFO] Lookup used (exact match): {test_raw_lookup.notna().sum()} rows")
print(f"[INFO] RF fallback:               {test_raw_lookup.isna().sum()} rows")

# ── 8. Export ──────────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "Index":  test["Index"],
    "demand": final_preds
})
submission.to_csv("submission_v2_lookup.csv", index=False)
print(f"\n[DONE] Saved: submission_v2_lookup.csv")
print(f"       Pred stats — min: {final_preds.min():.4f}, "
      f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}")