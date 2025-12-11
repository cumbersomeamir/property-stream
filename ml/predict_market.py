import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

base_dir = Path(__file__).parent.parent
models_dir = base_dir / 'ml' / 'models' / 'market'

# Load models and feature info
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

def predict(ward_features):
    """
    Predict market-based investment metrics for given ward features
    """
    # Create DataFrame with features
    feature_data = {}
    for col in feature_columns:
        feature_data[col] = [ward_features.get(col, 0)]
    
    X = pd.DataFrame(feature_data).fillna(0)
    
    # Make predictions
    predictions = {}
    for target_name, model in models.items():
        pred = model.predict(X)[0]
        predictions[target_name] = float(pred)
    
    # Calculate additional metrics
    investment_score = predictions['market_investment_score']
    rental_yield = predictions['market_rental_yield']
    appreciation_rate = predictions['market_appreciation_rate']
    
    # Calculate ROI for 3 years
    roi_3yr = ((1 + appreciation_rate / 100) ** 3 - 1) * 100
    
    # Risk score (inverse of investment score, normalized)
    risk_score = max(0, min(10, 10 - (investment_score / 10)))
    
    # Investment recommendation
    if investment_score >= 75:
        recommendation = "STRONG BUY"
        recommendation_reason = "Excellent growth potential, strong infrastructure, low risk - high returns expected"
    elif investment_score >= 65:
        recommendation = "BUY"
        recommendation_reason = "Good growth potential, decent infrastructure, moderate risk - positive returns expected"
    elif investment_score >= 55:
        recommendation = "HOLD"
        recommendation_reason = "Moderate potential, mixed signals, moderate risk - consider carefully"
    else:
        recommendation = "AVOID"
        recommendation_reason = "Lower growth potential, higher risk - consider alternative locations"
    
    return {
        'investment_score': round(investment_score, 2),
        'rental_yield': round(rental_yield, 2),
        'annual_appreciation_rate': round(appreciation_rate, 2),
        'roi_3yr': round(roi_3yr, 2),
        'risk_score': round(risk_score, 1),
        'recommendation': recommendation,
        'recommendation_reason': recommendation_reason,
        'model_type': 'market_based'
    }

if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    
    try:
        result = predict(input_data['features'])
        print(json.dumps({'success': True, 'predictions': result}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))

