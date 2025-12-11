import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False
    print("XGBoost not available, will skip it")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
    print("LightGBM not available, will skip it")

# Set up paths
base_dir = Path(__file__).parent.parent
data_dir = base_dir / 'data'

print("Loading and processing data...")

# Load all datasets
datasets = {}

def load_csv(filename):
    try:
        path = data_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            return df
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load datasets
datasets['demographics'] = load_csv('D01_DemographicProfile_Lucknow (1).csv')
datasets['unemployment'] = load_csv('D02_UnemploymentRate_Lucknow_2.csv')
datasets['households'] = load_csv('D03_Households_Lucknow_0.csv')
datasets['environment'] = load_csv('D04_Environment_lucknow_1.csv')
datasets['publicAmenities'] = load_csv('D05_PublicAmenities_lucknow_1.csv')
datasets['health'] = load_csv('D08_HealthInfrastructure_lucknow_1.csv')
datasets['propertyTax'] = load_csv('D21_PropertyTax_lucknow_1.csv')
datasets['waterTax'] = load_csv('D10_WaterTax_Lucknow_1_1.csv')
datasets['governance'] = load_csv('D44_Governance_Lucknow_1.csv')
datasets['streetLights'] = load_csv('D24_StreetLights_Lucknow_2.csv')
datasets['publicToilet'] = load_csv('D11_PublicToilet_Lucknow_2.csv')
datasets['solidWaste'] = load_csv('D12_D2D_Collection_Coverage.csv')
datasets['vehicles'] = load_csv('D40_VehicleRegistration_lucknow_1.csv')
datasets['buses'] = load_csv('D36_Buses_Lucknow.csv')
datasets['injuries'] = load_csv('D41_Injuries_Fatilities_Lucknow_1.csv')

print("Creating feature engineering pipeline...")

# Feature Engineering - Create comprehensive dataset at ward level
features_list = []

# Process households (ward level base)
if datasets['households'] is not None:
    for _, row in datasets['households'].iterrows():
        ward_data = {
            'ward_name': row.get('Ward Name', ''),
            'ward_no': row.get('Ward No', ''),
            'zone': str(row.get('Zone Name', '')).strip(),
            'households': pd.to_numeric(row.get('Total no of Households', 0), errors='coerce') or 0
        }
        features_list.append(ward_data)

features_df = pd.DataFrame(features_list)

# Extract zone number from zone name
def extract_zone_num(zone_str):
    if pd.isna(zone_str):
        return 0
    zone_str = str(zone_str).strip()
    for char in zone_str:
        if char.isdigit():
            return int(char)
    return 0

features_df['zone_num'] = features_df['zone'].apply(extract_zone_num)

# Aggregate zone-level features
zone_features = {}

# Property Tax Features
if datasets['propertyTax'] is not None:
    for _, row in datasets['propertyTax'].iterrows():
        zone = int(row.get('Zone Name', 0))
        if zone == 0:
            continue
            
        # Property tax growth rate
        tax_2013 = pd.to_numeric(row.get('2013-14 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
        tax_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
        tax_growth = ((tax_2017 - tax_2013) / tax_2013 * 100) if tax_2013 > 0 else 0
        
        # Commercial tax
        commercial_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Collection (in crores) - Commercial', 0), errors='coerce') or 0
        
        # Collection efficiency
        demand_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Demand (in crores) - Residential', 0), errors='coerce') or 0
        collection_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
        efficiency = (collection_2017 / demand_2017 * 100) if demand_2017 > 0 else 0
        
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone].update({
            'property_tax_growth_rate': tax_growth,
            'property_tax_2017': tax_2017,
            'commercial_tax_2017': commercial_2017,
            'tax_collection_efficiency': efficiency
        })

# Water Tax Features
if datasets['waterTax'] is not None:
    for _, row in datasets['waterTax'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0 or 'Govt' in zone_str:
            continue
            
        water_2013 = pd.to_numeric(row.get('Water tax collected (in INR lakhs)-Total-2013-14', 0), errors='coerce') or 0
        water_2017 = pd.to_numeric(row.get('Water tax collected (in INR lakhs)-Total-2017-18', 0), errors='coerce') or 0
        water_growth = ((water_2017 - water_2013) / water_2013 * 100) if water_2013 > 0 else 0
        
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone]['water_tax_growth'] = water_growth
        zone_features[zone]['water_tax_2017'] = water_2017

# Health Infrastructure
if datasets['health'] is not None:
    health_by_zone = {}
    for _, row in datasets['health'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0:
            continue
            
        if zone not in health_by_zone:
            health_by_zone[zone] = {'facilities': 0, 'total_beds': 0}
        
        beds = pd.to_numeric(row.get('Number of Beds in facility type', 0), errors='coerce') or 0
        if beds > 0:
            health_by_zone[zone]['facilities'] += 1
            health_by_zone[zone]['total_beds'] += beds
    
    for zone, stats in health_by_zone.items():
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone].update({
            'health_facilities': stats['facilities'],
            'health_beds': stats['total_beds']
        })

# Street Lights
if datasets['streetLights'] is not None:
    lights_by_zone = {}
    for _, row in datasets['streetLights'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0:
            continue
            
        poles = pd.to_numeric(row.get('Number of Poles', 0), errors='coerce') or 0
        if zone not in lights_by_zone:
            lights_by_zone[zone] = 0
        lights_by_zone[zone] += poles
    
    for zone, total_poles in lights_by_zone.items():
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone]['street_lights'] = total_poles

# Sewerage Coverage
if datasets['publicToilet'] is not None:
    for _, row in datasets['publicToilet'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0:
            continue
            
        households = pd.to_numeric(row.get('Total number of households (HH)', 0), errors='coerce') or 0
        sewerage = pd.to_numeric(row.get('HH part of the city sewerage network', 0), errors='coerce') or 0
        coverage = (sewerage / households * 100) if households > 0 else 0
        
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone]['sewerage_coverage'] = coverage

# Waste Collection Coverage
if datasets['solidWaste'] is not None:
    waste_by_zone = {}
    total_households_by_zone = {}
    covered_by_zone = {}
    
    for _, row in datasets['solidWaste'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0:
            continue
            
        households = pd.to_numeric(row.get('Total No. of households / establishments', 0), errors='coerce') or 0
        covered = pd.to_numeric(row.get('Total no. of households and establishments covered through doorstep collection', 0), errors='coerce') or 0
        
        if zone not in total_households_by_zone:
            total_households_by_zone[zone] = 0
            covered_by_zone[zone] = 0
        
        total_households_by_zone[zone] += households
        covered_by_zone[zone] += covered
    
    for zone in total_households_by_zone:
        coverage = (covered_by_zone[zone] / total_households_by_zone[zone] * 100) if total_households_by_zone[zone] > 0 else 0
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone]['waste_collection_coverage'] = coverage

# Governance - Voter Turnout
if datasets['governance'] is not None:
    governance_by_zone = {}
    for _, row in datasets['governance'].iterrows():
        zone_str = str(row.get('Zone Name', '')).strip()
        zone = extract_zone_num(zone_str)
        if zone == 0:
            continue
            
        registered = pd.to_numeric(row.get('Total no. of registered voters', 0), errors='coerce') or 0
        polled = pd.to_numeric(row.get('No. of votes polled in the last municipal election', 0), errors='coerce') or 0
        turnout = (polled / registered * 100) if registered > 0 else 0
        
        if zone not in governance_by_zone:
            governance_by_zone[zone] = []
        governance_by_zone[zone].append(turnout)
    
    for zone, turnouts in governance_by_zone.items():
        avg_turnout = np.mean(turnouts) if turnouts else 0
        if zone not in zone_features:
            zone_features[zone] = {}
        zone_features[zone]['avg_voter_turnout'] = avg_turnout

# Environment - Average PM2.5
if datasets['environment'] is not None:
    pm25_values = []
    for _, row in datasets['environment'].iterrows():
        pm25 = pd.to_numeric(row.get('Monthly mean/average concentration - PM2.5', 0), errors='coerce')
        if pd.notna(pm25) and pm25 > 0:
            pm25_values.append(pm25)
    
    avg_pm25 = np.mean(pm25_values) if pm25_values else 100
    # Apply same PM2.5 to all zones (city-level data)
    for zone in zone_features:
        zone_features[zone]['avg_pm25'] = avg_pm25

# Unemployment Rate (city-level)
if datasets['unemployment'] is not None and len(datasets['unemployment']) > 0:
    row = datasets['unemployment'].iloc[0]
    unemployed = pd.to_numeric(row.get('No. of unemployed persons (seeking or available for work)', 0), errors='coerce') or 0
    labor_force = pd.to_numeric(row.get('Total labour force in the city (age 15-59) [Employed + Unemployed Persons)', 0), errors='coerce') or 0
    unemployment_rate = (unemployed / labor_force * 100) if labor_force > 0 else 0
    
    for zone in zone_features:
        zone_features[zone]['unemployment_rate'] = unemployment_rate

# Merge zone features to ward-level data
print("Merging features...")

for idx, row in features_df.iterrows():
    zone = row['zone_num']
    if zone in zone_features:
        for key, value in zone_features[zone].items():
            features_df.at[idx, key] = value

# Fill NaN values with 0 or median
feature_columns = [col for col in features_df.columns if col not in ['ward_name', 'ward_no', 'zone', 'zone_num']]
for col in feature_columns:
    if col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

print(f"Total features: {len(feature_columns)}")
print(f"Feature columns: {feature_columns}")

# Create target variables (Investment Score based on multiple factors)
print("Creating target variables...")

def calculate_investment_score(row):
    score = 0
    
    # Property tax growth (0-25 points)
    tax_growth = row.get('property_tax_growth_rate', 0)
    score += min(25, max(0, tax_growth / 10))
    
    # Infrastructure (0-20 points)
    health_score = min(10, row.get('health_facilities', 0) / 5)
    lights_score = min(10, row.get('street_lights', 0) / 2000)
    score += health_score + lights_score
    
    # Services coverage (0-20 points)
    sewerage_score = min(10, row.get('sewerage_coverage', 0) / 10)
    waste_score = min(10, row.get('waste_collection_coverage', 0) / 10)
    score += sewerage_score + waste_score
    
    # Governance (0-15 points)
    turnout = row.get('avg_voter_turnout', 0)
    score += min(15, turnout / 5)
    
    # Efficiency (0-10 points)
    efficiency = row.get('tax_collection_efficiency', 0)
    score += min(10, efficiency / 10)
    
    # Commercial activity (0-10 points)
    commercial = row.get('commercial_tax_2017', 0)
    score += min(10, commercial / 10)
    
    # Environmental (negative impact) (0-10 points penalty)
    pm25 = row.get('avg_pm25', 100)
    env_penalty = max(0, (pm25 - 50) / 10)
    score -= min(10, env_penalty)
    
    return max(0, min(100, score))

features_df['investment_score'] = features_df.apply(calculate_investment_score, axis=1)

# Calculate Rental Yield (estimated based on features)
def calculate_rental_yield(row):
    base_yield = 4.0  # Base 4%
    
    # Adjust based on infrastructure
    if row.get('health_facilities', 0) > 10:
        base_yield += 0.5
    if row.get('street_lights', 0) > 5000:
        base_yield += 0.3
    if row.get('sewerage_coverage', 0) > 70:
        base_yield += 0.5
    if row.get('waste_collection_coverage', 0) > 90:
        base_yield += 0.2
    
    # Adjust based on commercial activity
    if row.get('commercial_tax_2017', 0) > 50:
        base_yield += 0.5
    
    # Adjust based on growth
    if row.get('property_tax_growth_rate', 0) > 20:
        base_yield += 0.5
    
    return min(10.0, max(3.0, base_yield))

features_df['rental_yield'] = features_df.apply(calculate_rental_yield, axis=1)

# Calculate Appreciation Rate (annual %)
def calculate_appreciation_rate(row):
    base_appreciation = 8.0  # Base 8% annual
    
    # Growth factors
    tax_growth = row.get('property_tax_growth_rate', 0)
    base_appreciation += min(5, tax_growth / 5)
    
    # Infrastructure premium
    if row.get('health_facilities', 0) > 15:
        base_appreciation += 1.5
    if row.get('street_lights', 0) > 6000:
        base_appreciation += 1.0
    
    # Commercial premium
    if row.get('commercial_tax_2017', 0) > 100:
        base_appreciation += 2.0
    
    # Governance premium
    if row.get('avg_voter_turnout', 0) > 40:
        base_appreciation += 1.0
    
    return min(20.0, max(5.0, base_appreciation))

features_df['annual_appreciation_rate'] = features_df.apply(calculate_appreciation_rate, axis=1)

print(f"\nDataset shape: {features_df.shape}")
print(f"Investment Score range: {features_df['investment_score'].min():.2f} - {features_df['investment_score'].max():.2f}")
print(f"Rental Yield range: {features_df['rental_yield'].min():.2f}% - {features_df['rental_yield'].max():.2f}%")
print(f"Appreciation Rate range: {features_df['annual_appreciation_rate'].min():.2f}% - {features_df['annual_appreciation_rate'].max():.2f}%")

# Prepare data for modeling
X = features_df[feature_columns].fillna(0)
targets = {
    'investment_score': features_df['investment_score'],
    'rental_yield': features_df['rental_yield'],
    'annual_appreciation_rate': features_df['annual_appreciation_rate']
}

# Split data
X_train, X_test, idx_train, idx_test = train_test_split(X, features_df.index, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train models for each target
models = {}
results = {}

for target_name, y in targets.items():
    print(f"\n{'='*60}")
    print(f"Training models for: {target_name}")
    print(f"{'='*60}")
    
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]
    
    target_models = {}
    target_results = {}
    
    # 1. Random Forest
    print("\n1. Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    target_models['random_forest'] = rf
    target_results['random_forest'] = {
        'mse': rf_mse,
        'rmse': np.sqrt(rf_mse),
        'mae': rf_mae,
        'r2': rf_r2
    }
    print(f"   RMSE: {np.sqrt(rf_mse):.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
    
    # 2. Gradient Boosting
    print("\n2. Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    
    target_models['gradient_boosting'] = gb
    target_results['gradient_boosting'] = {
        'mse': gb_mse,
        'rmse': np.sqrt(gb_mse),
        'mae': gb_mae,
        'r2': gb_r2
    }
    print(f"   RMSE: {np.sqrt(gb_mse):.4f}, MAE: {gb_mae:.4f}, R²: {gb_r2:.4f}")
    
    # 3. XGBoost
    if HAS_XGBOOST:
        print("\n3. Training XGBoost...")
        try:
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            xgb_mse = mean_squared_error(y_test, xgb_pred)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
            
            target_models['xgboost'] = xgb_model
            target_results['xgboost'] = {
                'mse': xgb_mse,
                'rmse': np.sqrt(xgb_mse),
                'mae': xgb_mae,
                'r2': xgb_r2
            }
            print(f"   RMSE: {np.sqrt(xgb_mse):.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}")
        except Exception as e:
            print(f"   XGBoost failed: {e}")
    else:
        print("\n3. XGBoost skipped (not available)")
    
    # 4. LightGBM
    if HAS_LIGHTGBM:
        print("\n4. Training LightGBM...")
        try:
            lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            
            lgb_mse = mean_squared_error(y_test, lgb_pred)
            lgb_mae = mean_absolute_error(y_test, lgb_pred)
            lgb_r2 = r2_score(y_test, lgb_pred)
            
            target_models['lightgbm'] = lgb_model
            target_results['lightgbm'] = {
                'mse': lgb_mse,
                'rmse': np.sqrt(lgb_mse),
                'mae': lgb_mae,
                'r2': lgb_r2
            }
            print(f"   RMSE: {np.sqrt(lgb_mse):.4f}, MAE: {lgb_mae:.4f}, R²: {lgb_r2:.4f}")
        except Exception as e:
            print(f"   LightGBM failed: {e}")
    else:
        print("\n4. LightGBM skipped (not available)")
    
    # Cross-validation for more robust metrics
    print("\n5. Running Cross-Validation...")
    cv_scores_r2 = {}
    cv_scores_rmse = {}
    
    for model_name, model in target_models.items():
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
        
        cv_scores_r2[model_name] = cv_r2.mean()
        cv_scores_rmse[model_name] = cv_rmse_scores.mean()
        print(f"   {model_name}: CV R² = {cv_r2.mean():.4f} (±{cv_r2.std():.4f}), CV RMSE = {cv_rmse_scores.mean():.4f}")
    
    # Select best model based on test performance
    best_model_name = max(target_results.keys(), key=lambda k: target_results[k]['r2'])
    best_model = target_models[best_model_name]
    best_results = target_results[best_model_name]
    
    print(f"\n✓ Best model for {target_name}: {best_model_name}")
    print(f"  Test RMSE: {best_results['rmse']:.4f}")
    print(f"  Test MAE: {best_results['mae']:.4f}")
    print(f"  Test R² Score: {best_results['r2']:.4f}")
    print(f"  CV R² Score: {cv_scores_r2[best_model_name]:.4f} (±{np.std([cv_scores_r2[k] for k in cv_scores_r2 if k == best_model_name]):.4f})")
    print(f"  CV RMSE: {cv_scores_rmse[best_model_name]:.4f}")
    
    if y_test.std() > 0:
        test_accuracy = (1 - best_results['rmse'] / y_test.std()) * 100
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
    
    models[target_name] = {
        'model': best_model,
        'model_name': best_model_name,
        'metrics': best_results,
        'cv_metrics': {
            'r2_mean': cv_scores_r2[best_model_name],
            'r2_std': np.std([cv_scores_r2[k] for k in cv_scores_r2 if k == best_model_name]),
            'rmse_mean': cv_scores_rmse[best_model_name]
        },
        'all_models': target_models,
        'all_results': target_results
    }
    results[target_name] = {
        **best_results,
        'cv_r2': cv_scores_r2[best_model_name],
        'cv_rmse': cv_scores_rmse[best_model_name]
    }

# Save models
models_dir = base_dir / 'ml' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("Saving models...")
print(f"{'='*60}")

for target_name, model_info in models.items():
    model_path = models_dir / f'{target_name}_model.pkl'
    joblib.dump(model_info['model'], model_path)
    print(f"Saved: {model_path}")

# Save feature columns and scaler
feature_info = {
    'feature_columns': feature_columns,
    'scaler': None  # Not using scaler for tree-based models
}
joblib.dump(feature_info, models_dir / 'feature_info.pkl')

# Save metadata
metadata = {
    'results': results,
    'model_info': {k: {'name': v['model_name'], 'metrics': v['metrics']} for k, v in models.items()},
    'train_size': len(X_train),
    'test_size': len(X_test)
}
joblib.dump(metadata, models_dir / 'model_metadata.pkl')

# Print summary
print(f"\n{'='*60}")
print("MODEL TRAINING SUMMARY")
print(f"{'='*60}")
print(f"\nDataset: {len(features_df)} wards")
print(f"Features: {len(feature_columns)}")
print(f"Train/Test split: {len(X_train)} / {len(X_test)}")

for target_name, result in results.items():
    print(f"\n{target_name.upper().replace('_', ' ')}:")
    print(f"  Best Model: {models[target_name]['model_name']}")
    print(f"  Test R² Score: {result['r2']:.4f}")
    print(f"  Test RMSE: {result['rmse']:.4f}")
    print(f"  Test MAE: {result['mae']:.4f}")
    print(f"  CV R² Score: {result.get('cv_r2', 0):.4f}")
    print(f"  CV RMSE: {result.get('cv_rmse', 0):.4f}")
    if targets[target_name].std() > 0:
        accuracy = (1 - result['rmse'] / targets[target_name].std()) * 100
        print(f"  Test Accuracy: {accuracy:.2f}%")

print(f"\n{'='*60}")
print("Training complete! Models saved to ml/models/")
print(f"{'='*60}")

