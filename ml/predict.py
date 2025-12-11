import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

base_dir = Path(__file__).parent.parent
models_dir = base_dir / 'ml' / 'models'

# Load models and feature info
feature_info = joblib.load(models_dir / 'feature_info.pkl')
feature_columns = feature_info['feature_columns']

models = {
    'investment_score': joblib.load(models_dir / 'investment_score_model.pkl'),
    'rental_yield': joblib.load(models_dir / 'rental_yield_model.pkl'),
    'annual_appreciation_rate': joblib.load(models_dir / 'annual_appreciation_rate_model.pkl')
}

def predict(ward_features):
    """
    Predict investment metrics for given ward features
    ward_features: dict with feature values
    """
    # Create DataFrame with features
    feature_data = {}
    for col in feature_columns:
        feature_data[col] = [ward_features.get(col, 0)]
    
    X = pd.DataFrame(feature_data)
    
    # Make predictions
    predictions = {}
    for target_name, model in models.items():
        pred = model.predict(X)[0]
        predictions[target_name] = float(pred)
    
    # Calculate additional metrics
    investment_score = predictions['investment_score']
    rental_yield = predictions['rental_yield']
    appreciation_rate = predictions['annual_appreciation_rate']
    
    # Calculate ROI for 3 years
    roi_3yr = ((1 + appreciation_rate / 100) ** 3 - 1) * 100
    
    # Risk score (inverse of investment score, normalized)
    risk_score = max(0, min(10, 10 - (investment_score / 10)))
    
    # Investment recommendation
    if investment_score >= 45:
        recommendation = "STRONG BUY"
        recommendation_reason = "Excellent infrastructure, high growth potential, strong governance"
    elif investment_score >= 35:
        recommendation = "BUY"
        recommendation_reason = "Good infrastructure, positive growth trends, decent governance"
    elif investment_score >= 25:
        recommendation = "HOLD"
        recommendation_reason = "Moderate potential, mixed signals, average infrastructure"
    else:
        recommendation = "AVOID"
        recommendation_reason = "Low infrastructure, poor growth trends, high risk"
    
    return {
        'investment_score': round(investment_score, 2),
        'rental_yield': round(rental_yield, 2),
        'annual_appreciation_rate': round(appreciation_rate, 2),
        'roi_3yr': round(roi_3yr, 2),
        'risk_score': round(risk_score, 1),
        'recommendation': recommendation,
        'recommendation_reason': recommendation_reason
    }

if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    try:
        result = predict(input_data['features'])
        print(json.dumps({'success': True, 'predictions': result}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))

