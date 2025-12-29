import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

base_dir = Path(__file__).parent.parent
models_dir = base_dir / 'ml' / 'models' / 'market'
data_dir = base_dir / 'data'

# Load models
feature_info = joblib.load(models_dir / 'feature_info.pkl')
feature_columns = feature_info['feature_columns']

models = {
    'market_appreciation_rate': joblib.load(models_dir / 'market_appreciation_rate_model.pkl'),
    'market_rental_yield': joblib.load(models_dir / 'market_rental_yield_model.pkl'),
    'market_investment_score': joblib.load(models_dir / 'market_investment_score_model.pkl')
}

def extract_zone_num(zone_str):
    if pd.isna(zone_str):
        return 0
    zone_str = str(zone_str).strip()
    for char in zone_str:
        if char.isdigit():
            return int(char)
    return 0

# Load all datasets
households_df = pd.read_csv(data_dir / 'D03_Households_Lucknow_0.csv')
property_tax_df = pd.read_csv(data_dir / 'D21_PropertyTax_lucknow_1.csv')
water_tax_df = pd.read_csv(data_dir / 'D10_WaterTax_Lucknow_1_1.csv')
health_df = pd.read_csv(data_dir / 'D08_HealthInfrastructure_lucknow_1.csv')
street_lights_df = pd.read_csv(data_dir / 'D24_StreetLights_Lucknow_2.csv')
public_toilet_df = pd.read_csv(data_dir / 'D11_PublicToilet_Lucknow_2.csv')
solid_waste_df = pd.read_csv(data_dir / 'D12_D2D_Collection_Coverage.csv')
governance_df = pd.read_csv(data_dir / 'D44_Governance_Lucknow_1.csv')
unemployment_df = pd.read_csv(data_dir / 'D02_UnemploymentRate_Lucknow_2.csv')

# Pre-process zone-level data
zone_features_map = {}

# Property Tax - calculate CAGR and volatility
for _, row in property_tax_df.iterrows():
    zone = int(row.get('Zone Name', 0))
    if zone == 0:
        continue
    
    tax_2013 = pd.to_numeric(row.get('2013-14 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
    tax_2014 = pd.to_numeric(row.get('2014-15 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
    tax_2015 = pd.to_numeric(row.get('2015-16 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
    tax_2016 = pd.to_numeric(row.get('2016-17 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
    tax_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Collection (in crores) - Residential', 0), errors='coerce') or 0
    
    avg_tax = (tax_2013 + tax_2014 + tax_2015 + tax_2016 + tax_2017) / 5
    years = 4
    cagr = ((tax_2017 / tax_2013) ** (1/years) - 1) * 100 if tax_2013 > 0 else 0
    
    taxes = [tax_2013, tax_2014, tax_2015, tax_2016, tax_2017]
    taxes_clean = [t for t in taxes if t > 0]
    volatility = np.std(taxes_clean) / np.mean(taxes_clean) * 100 if len(taxes_clean) > 1 and np.mean(taxes_clean) > 0 else 0
    
    commercial_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Collection (in crores) - Commercial', 0), errors='coerce') or 0
    demand_2017 = pd.to_numeric(row.get('2017-18 - Property Tax Demand (in crores) - Residential', 0), errors='coerce') or 0
    collection_efficiency = (tax_2017 / demand_2017 * 100) if demand_2017 > 0 else 0
    
    if zone not in zone_features_map:
        zone_features_map[zone] = {}
    zone_features_map[zone].update({
        'property_tax_cagr': cagr,
        'avg_property_tax': avg_tax,
        'tax_volatility': volatility,
        'commercial_tax_2017': commercial_2017,
        'tax_collection_efficiency': collection_efficiency,
        'recent_tax_growth': ((tax_2017 - tax_2015) / tax_2015 * 100) if tax_2015 > 0 else 0
    })

# Water Tax
for _, row in water_tax_df.iterrows():
    zone = extract_zone_num(str(row.get('Zone Name', '')))
    if zone == 0 or 'Govt' in str(row.get('Zone Name', '')):
        continue
    water_2013 = pd.to_numeric(row.get('Water tax collected (in INR lakhs)-Total-2013-14', 0), errors='coerce') or 0
    water_2017 = pd.to_numeric(row.get('Water tax collected (in INR lakhs)-Total-2017-18', 0), errors='coerce') or 0
    water_cagr = ((water_2017 / water_2013) ** (1/4) - 1) * 100 if water_2013 > 0 else 0
    
    if zone not in zone_features_map:
        zone_features_map[zone] = {}
    zone_features_map[zone]['water_tax_cagr'] = water_cagr

# Health
health_by_zone = {}
for _, row in health_df.iterrows():
    zone = extract_zone_num(str(row.get('Zone Name', '')))
    if zone == 0:
        continue
    if zone not in health_by_zone:
        health_by_zone[zone] = {'facilities': 0, 'total_beds': 0}
    beds = pd.to_numeric(row.get('Number of Beds in facility type', 0), errors='coerce') or 0
    if beds > 0:
        health_by_zone[zone]['facilities'] += 1
        health_by_zone[zone]['total_beds'] += beds

for zone, stats in health_by_zone.items():
    if zone not in zone_features_map:
        zone_features_map[zone] = {}
    zone_features_map[zone].update({
        'health_facilities': stats['facilities'],
        'health_beds': stats['total_beds']
    })

# Street Lights
lights_by_zone = {}
for _, row in street_lights_df.iterrows():
    zone = extract_zone_num(str(row.get('Zone Name', '')))
    if zone == 0:
        continue
    poles = pd.to_numeric(row.get('Number of Poles', 0), errors='coerce') or 0
    if zone not in lights_by_zone:
        lights_by_zone[zone] = 0
    lights_by_zone[zone] += poles

for zone, total_poles in lights_by_zone.items():
    if zone not in zone_features_map:
        zone_features_map[zone] = {}
    zone_features_map[zone]['street_lights'] = total_poles

# Sewerage
sewerage_by_zone = {}
for _, row in public_toilet_df.iterrows():
    zone = extract_zone_num(str(row.get('Zone Name', '')))
    if zone == 0:
        continue
    households = pd.to_numeric(row.get('Total number of households (HH)', 0), errors='coerce') or 0
    sewerage = pd.to_numeric(row.get('HH part of the city sewerage network', 0), errors='coerce') or 0
    coverage = (sewerage / households * 100) if households > 0 else 0
    
    if zone not in zone_features_map:
        zone_features_map[zone] = {}
    zone_features_map[zone]['sewerage_coverage'] = coverage

# Waste Collection by ward
waste_by_ward = {}
for _, row in solid_waste_df.iterrows():
    ward_name = str(row.get('Ward Name', ''))
    households = pd.to_numeric(row.get('Total No. of households / establishments', 0), errors='coerce') or 0
    covered = pd.to_numeric(row.get('Total no. of households and establishments covered through doorstep collection', 0), errors='coerce') or 0
    coverage = (covered / households * 100) if households > 0 else 0
    waste_by_ward[ward_name] = coverage

# Governance by ward
governance_by_ward = {}
for _, row in governance_df.iterrows():
    ward_name = str(row.get('Ward Name', ''))
    registered = pd.to_numeric(row.get('Total no. of registered voters', 0), errors='coerce') or 0
    polled = pd.to_numeric(row.get('No. of votes polled in the last municipal election', 0), errors='coerce') or 0
    turnout = (polled / registered * 100) if registered > 0 else 0
    if ward_name not in governance_by_ward:
        governance_by_ward[ward_name] = []
    governance_by_ward[ward_name].append(turnout)

for ward_name, turnouts in governance_by_ward.items():
    governance_by_ward[ward_name] = np.mean(turnouts) if turnouts else 0

# Unemployment rate
unemployment_rate = 9.93
if len(unemployment_df) > 0:
    row = unemployment_df.iloc[0]
    unemployed = pd.to_numeric(row.get('No. of unemployed persons (seeking or available for work)', 0), errors='coerce') or 0
    labor_force = pd.to_numeric(row.get('Total labour force in the city (age 15-59) [Employed + Unemployed Persons)', 0), errors='coerce') or 0
    unemployment_rate = (unemployed / labor_force * 100) if labor_force > 0 else 9.93

# Generate predictions for all wards
predictions = []

for _, ward_row in households_df.iterrows():
    ward_name = str(ward_row.get('Ward Name', ''))
    ward_no = ward_row.get('Ward No', '')
    zone_str = str(ward_row.get('Zone Name', '')).strip()
    zone = extract_zone_num(zone_str)
    households = pd.to_numeric(ward_row.get('Total no of Households', 0), errors='coerce') or 0
    
    # Build feature vector using market model features
    features = {
        'households': households,
        'property_tax_cagr': zone_features_map.get(zone, {}).get('property_tax_cagr', 0),
        'avg_property_tax': zone_features_map.get(zone, {}).get('avg_property_tax', 0),
        'tax_volatility': zone_features_map.get(zone, {}).get('tax_volatility', 0),
        'commercial_tax_2017': zone_features_map.get(zone, {}).get('commercial_tax_2017', 0),
        'tax_collection_efficiency': zone_features_map.get(zone, {}).get('tax_collection_efficiency', 0),
        'recent_tax_growth': zone_features_map.get(zone, {}).get('recent_tax_growth', 0),
        'water_tax_cagr': zone_features_map.get(zone, {}).get('water_tax_cagr', 0),
        'health_facilities': zone_features_map.get(zone, {}).get('health_facilities', 0),
        'health_beds': zone_features_map.get(zone, {}).get('health_beds', 0),
        'street_lights': zone_features_map.get(zone, {}).get('street_lights', 0),
        'sewerage_coverage': zone_features_map.get(zone, {}).get('sewerage_coverage', 0),
        'waste_collection_coverage': waste_by_ward.get(ward_name, 0),
        'avg_voter_turnout': governance_by_ward.get(ward_name, 0),
        'unemployment_rate': unemployment_rate
    }
    
    # Create DataFrame for prediction
    feature_data = {col: [features.get(col, 0)] for col in feature_columns}
    X = pd.DataFrame(feature_data).fillna(0)
    
    # Make predictions
    investment_score = float(models['market_investment_score'].predict(X)[0])
    rental_yield = float(models['market_rental_yield'].predict(X)[0])
    appreciation_rate = float(models['market_appreciation_rate'].predict(X)[0])
    
    # Calculate additional metrics
    roi_3yr = ((1 + appreciation_rate / 100) ** 3 - 1) * 100
    risk_score = max(0, min(10, 10 - (investment_score / 10)))
    
    # Recommendation
    if investment_score >= 75:
        recommendation = "STRONG BUY"
    elif investment_score >= 65:
        recommendation = "BUY"
    elif investment_score >= 55:
        recommendation = "HOLD"
    else:
        recommendation = "AVOID"
    
    predictions.append({
        'ward_name': ward_name,
        'ward_no': str(ward_no),
        'zone': zone_str,
        'zone_num': zone,
        'households': int(households),
        'investment_score': round(investment_score, 2),
        'rental_yield': round(rental_yield, 2),
        'annual_appreciation_rate': round(appreciation_rate, 2),
        'roi_3yr': round(roi_3yr, 2),
        'risk_score': round(risk_score, 1),
        'recommendation': recommendation
    })

# Sort by investment score
predictions.sort(key=lambda x: x['investment_score'], reverse=True)

# Add ranking
for idx, pred in enumerate(predictions, 1):
    pred['rank'] = idx

# Output JSON
output = {
    'success': True,
    'total_wards': len(predictions),
    'model_type': 'market_based',
    'predictions': predictions
}

print(json.dumps(output, indent=2))


