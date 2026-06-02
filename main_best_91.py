#!/usr/bin/env python3
"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction Model
Optimized Machine Learning Pipeline
Target Metric: R2 Score (R-squared)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PURE-PYTHON GEOHASH DECODER (Zero external dependencies)
# -----------------------------------------------------------------------------
def decode_geohash(geohash_str):
    """
    Decodes a geohash string into latitude and longitude coordinates.
    Returns: (latitude, longitude) midpoints of the geohash grid bounding box.
    """
    if not isinstance(geohash_str, str) or len(geohash_str) == 0:
        return np.nan, np.nan
        
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    decoder = {char: i for i, char in enumerate(base32)}
    
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    
    for char in geohash_str.lower():
        if char not in decoder:
            continue
        val = decoder[char]
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                # Longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if val & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                # Latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if val & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING ENGINE
# -----------------------------------------------------------------------------
def process_data(train_path, test_path):
    """
    Loads, cleans, and engineers robust spatial/temporal features for training and test data.
    """
    print("📂 Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Save the original test indices for the submission file
    test_indices = test['Index'].copy()
    
    print(f"📊 Train Shape: {train.shape} | Test Shape: {test.shape}")
    
    # Concatenate for uniform feature engineering
    target_col = 'demand'
    train[target_col] = train[target_col].astype(float)
    y_train = train[target_col].values
    
    # Keep track of training rows
    n_train = len(train)
    
    # Combine datasets to ensure identical features and categorical mappings
    combined = pd.concat([train.drop(columns=[target_col]), test], axis=0).reset_index(drop=True)
    
    print("🗺️  1/4 Decoding Geohashes...")
    # Decode latitude and longitude from spatial geohash codes
    lat_lon = combined['geohash'].apply(decode_geohash)
    combined['lat'] = [x[0] for x in lat_lon]
    combined['lon'] = [x[1] for x in lat_lon]
    
    # Calculate simple distance from center of coordinates (urban center approximation)
    lat_mean = combined['lat'].mean()
    lon_mean = combined['lon'].mean()
    combined['dist_from_center'] = np.sqrt((combined['lat'] - lat_mean)**2 + (combined['lon'] - lon_mean)**2)
    
    # Geohash prefixes capture regional hierarchies
    combined['geohash_prefix_3'] = combined['geohash'].astype(str).str[:3]
    combined['geohash_prefix_4'] = combined['geohash'].astype(str).str[:4]
    
    # Spatial density features (how often does a geohash occur in the dataset)
    geohash_freq = combined['geohash'].value_counts().to_dict()
    combined['geohash_frequency'] = combined['geohash'].map(geohash_freq)
    
    print("⏰ 2/4 Parsing Temporal Fields...")
    # Parse timestamp (HH:MM or standard formats)
    if 'timestamp' in combined.columns:
        str_times = combined['timestamp'].astype(str).str.split(':', expand=True)
        if str_times.shape[1] >= 2:
            hours_str = str_times[0]
            minutes_str = str_times[1]
        else:
            hours_str = pd.Series([np.nan] * len(combined))
            minutes_str = pd.Series([np.nan] * len(combined))
            
        ts_dt = pd.to_datetime(combined['timestamp'], errors='coerce')
        hours = ts_dt.dt.hour.fillna(pd.to_numeric(hours_str, errors='coerce'))
        minutes = ts_dt.dt.minute.fillna(pd.to_numeric(minutes_str, errors='coerce'))
        
        combined['hour'] = hours.fillna(12).astype(int)
        combined['minute'] = minutes.fillna(0).astype(int)
    else:
        combined['hour'] = 12
        combined['minute'] = 0
        
    # Cyclical encoding of hours (representing that 23:00 is close to 01:00)
    combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24.0)
    combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24.0)
    combined['time_of_day_min'] = combined['hour'] * 60 + combined['minute']
    
    # Process "day" variable
    if 'day' in combined.columns:
        combined['day_numeric'] = pd.to_numeric(combined['day'], errors='coerce')
        if combined['day_numeric'].isna().sum() < len(combined) * 0.5:
            # Mostly numeric days (e.g., day of week or day of month)
            combined['day_numeric'] = combined['day_numeric'].fillna(-1)
            # Weekend estimation
            combined['is_weekend'] = combined['day_numeric'].isin([6, 7, 13, 14, 20, 21, 27, 28]).astype(int)
        else:
            # Categorical weekday names
            combined['is_weekend'] = combined['day'].astype(str).str.lower().str.contains('sat|sun|weekend|6|7').astype(int)
            combined['day_numeric'] = combined['day'].factorize()[0]
    else:
        combined['is_weekend'] = 0
        combined['day_numeric'] = 0
        
    print("🛠️  3/4 Engineering Infrastructure & Weather Features...")
    # Fill missing lanes with median
    combined['NumberofLanes'] = combined['NumberofLanes'].fillna(combined['NumberofLanes'].median())
    
    # Fill continuous temperature with median, create missingness flag
    combined['Temperature_isna'] = combined['Temperature'].isna().astype(int)
    combined['Temperature'] = combined['Temperature'].fillna(combined['Temperature'].median())
    
    # Categorical Columns to Encode
    categorical_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geohash_prefix_3', 'geohash_prefix_4']
    for col in categorical_cols:
        combined[col] = combined[col].astype(str).fillna('Missing')
        # Label Encoder
        le = LabelEncoder()
        combined[f'{col}_encoded'] = le.fit_transform(combined[col])
        
    # Drop raw and redundant columns to prepare for model input
    drop_cols = ['Index', 'geohash', 'timestamp', 'day'] + categorical_cols
    df_features = combined.drop(columns=drop_cols, errors='ignore')
    
    # Split back to Train and Test sets
    X_train = df_features.iloc[:n_train].copy()
    X_test = df_features.iloc[n_train:].copy()
    
    # Re-attach target variable to training set for OOF target encoding
    X_train[target_col] = y_train
    # Standard temporary categorical features for OOF target encoding
    X_train['geohash'] = train['geohash'].values
    X_test['geohash'] = test['geohash'].values
    
    print("🎯 4/4 Building Safe Out-Of-Fold (OOF) Target Encoding...")
    # Safe Target Encoding to prevent spatial feature leakage
    target_enc_cols = ['geohash']
    for col in target_enc_cols:
        col_name = f'{col}_target_enc'
        X_train[col_name] = np.nan
        X_test[col_name] = np.nan
        
        global_mean = y_train.mean()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for trn_idx, val_idx in kf.split(X_train):
            trn_fold = X_train.iloc[trn_idx]
            val_fold = X_train.iloc[val_idx]
            
            # Smooth category means
            cat_means = trn_fold.groupby(col)[target_col].mean()
            cat_counts = trn_fold.groupby(col)[target_col].count()
            smooth_weight = 10.0
            smoothed_means = (cat_means * cat_counts + global_mean * smooth_weight) / (cat_counts + smooth_weight)
            
            X_train.iloc[val_idx, X_train.columns.get_loc(col_name)] = val_fold[col].map(smoothed_means).fillna(global_mean)
            
        # Complete train mapping for the test set
        full_means = X_train.groupby(col)[target_col].mean()
        full_counts = X_train.groupby(col)[target_col].count()
        smoothed_full_means = (full_means * full_counts + global_mean * smooth_weight) / (full_counts + smooth_weight)
        X_test[col_name] = X_test[col].map(smoothed_full_means).fillna(global_mean)
        
    # Drop standard categorical and targets from models features
    X_train.drop(columns=[target_col, 'geohash'], inplace=True)
    X_test.drop(columns=['geohash'], inplace=True)
    
    # Handle any final missing values
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
        
    return X_train, y_train, X_test, test_indices

# -----------------------------------------------------------------------------
# 3. TRAINING ENGINE WITH AUTO-FALLBACK
# -----------------------------------------------------------------------------
def train_and_predict(X_train, y_train, X_test, folds=5):
    """
    Trains multiple boosting models using Cross-Validation and generates robust ensembled predictions.
    Includes auto-fallback to alternate regressor models based on installed system libraries.
    """
    # 3a. Model Selection & Library Imports
    model_type = None
    
    try:
        import lightgbm as lgb
        model_type = 'lightgbm'
        print("⚡ LightGBM imported successfully. Running LightGBM pipeline...")
    except ImportError:
        try:
            import xgboost as xgb
            model_type = 'xgboost'
            print("🚀 XGBoost imported successfully. Running XGBoost pipeline...")
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                model_type = 'hist_gb'
                print("🌲 Scikit-Learn HistGradientBoostingRegressor imported successfully...")
            except ImportError:
                from sklearn.ensemble import RandomForestRegressor
                model_type = 'rf'
                print("🌳 Fallback to RandomForestRegressor...")
                
    # 3b. Setup Cross-Validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    
    print(f"🤖 Training using: {model_type.upper()}")
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        if model_type == 'lightgbm':
            import lightgbm as lgb
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 63,
                'max_depth': 8,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'random_state': 42 + fold,
                'verbose': -1,
                'n_jobs': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
            
            val_pred = model.predict(X_va, num_iteration=model.best_iteration)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
        elif model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                n_jobs=-1,
                early_stopping_rounds=50
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            val_pred = model.predict(X_va)
            test_pred = model.predict(X_test)
            
        elif model_type == 'hist_gb':
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=8,
                learning_rate=0.07,
                random_state=42 + fold
            )
            model.fit(X_tr, y_tr)
            
            val_pred = model.predict(X_va)
            test_pred = model.predict(X_test)
            
        else: # RandomForest Fallback
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                n_jobs=-1,
                random_state=42 + fold
            )
            model.fit(X_tr, y_tr)
            
            val_pred = model.predict(X_va)
            test_pred = model.predict(X_test)
            
        # Save validation and test predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / folds
        
        # Calculate local validation R2 score for fold
        fold_r2 = r2_score(y_va, val_pred)
        print(f"   👉 Fold {fold + 1} R2 Score: {fold_r2:.5f}")
        
    overall_r2 = r2_score(y_train, oof_predictions)
    score_hackerearth = max(0, 100 * overall_r2)
    print(f"\n🏆 Overall Out-Of-Fold R2 Score: {overall_r2:.6f}")
    print(f"🔥 Expected HackerEarth Score: {score_hackerearth:.2f} / 100.0")
    
    return test_predictions

# -----------------------------------------------------------------------------
# 4. MAIN FLOW CONTROL
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🚦 STARTING TRAFFIC DEMAND PREDICTION ML PIPELINE 🚦")
    print("=" * 60)
    
    train_path = 'train.csv'
    test_path = 'test.csv'
    sample_sub_path = 'sample_submission.csv'
    output_path = 'submission.csv'

    
    # 4a. Check dataset files
    missing_files = []
    for f in [train_path, test_path]:
        if not os.path.exists(f):
            missing_files.append(f)
            
    if missing_files:
        print(f"❌ Error: Required files {missing_files} not found in this directory.")
        print("Please copy 'train.csv' and 'test.csv' to this directory before running this script.")
        sys.exit(1)
        
    # 4b. Run Pipeline
    try:
        # Load and engineer features
        X_train, y_train, X_test, test_indices = process_data(train_path, test_path)
        
        # Train and generate predictions
        predictions = train_and_predict(X_train, y_train, X_test)
        
        # 4c. Generate Submission
        print("\n📝 Saving submission predictions...")
        submission = pd.DataFrame({
            'Index': test_indices,
            'demand': predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"✅ Submission saved to: {output_path}")
        
        # 4d. Verification checks
        print("\n🔍 Verifying submission file structure...")
        sub_df = pd.read_csv(output_path)
        print(f"   👉 Submission Shape: {sub_df.shape} (Expected: 41778, 2)")
        print(f"   👉 Index Column Matches Test: {sub_df['Index'].equals(test_indices)}")
        print(f"   👉 Head of submission.csv:")
        print(sub_df.head())
        print("\n🚀 Model training and validation checks passed! Ready for upload to HackerEarth!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Critical Pipeline Failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()