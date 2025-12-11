import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

base_dir = Path(__file__).parent.parent
models_dir = base_dir / 'ml' / 'models' / 'commercial_growth'
data_dir = base_dir / 'data'

# Load models and metadata
metadata = joblib.load(models_dir / 'model_metadata.pkl')
feature_columns = metadata['feature_columns']
model1 = joblib.load(models_dir / 'model1_gradient_boosting.pkl')

try:
    import tensorflow as tf
    model2 = tf.keras.models.load_model(str(models_dir / 'model2_dnn.keras'))
    scaler = joblib.load(models_dir / 'dnn_scaler.pkl')
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorFlow model not available: {e}", file=__import__('sys').stderr)
    TENSORFLOW_AVAILABLE = False

# Load master dataset
master_df = pd.read_csv(models_dir / 'master_dataset.csv')

# Make predictions
predictions = []

for _, row in master_df.iterrows():
    ward_name = str(row.get('Ward Name', ''))
    ward_no = str(row.get('Ward No', ''))
    zone = str(row.get('Zone Name', ''))
    
    # Extract features
    feature_values = [pd.to_numeric(row.get(col, 0), errors='coerce') or 0 for col in feature_columns]
    features_df = pd.DataFrame([feature_values], columns=feature_columns)
    
    # Model 1 prediction
    pred1 = float(model1.predict(features_df)[0])
    
    # Model 2 prediction
    pred2 = None
    if TENSORFLOW_AVAILABLE:
        try:
            features_scaled = scaler.transform(features_df)
            pred2 = float(model2.predict(features_scaled, verbose=0)[0][0])
            # Ensemble: average of both models
            final_pred = (pred1 + pred2) / 2
        except Exception as e:
            final_pred = pred1
    else:
        final_pred = pred1
    
    predictions.append({
        'ward_name': ward_name,
        'ward_no': str(ward_no),
        'zone': zone,
        'commercial_growth_rate': round(final_pred, 2),
        'model1_prediction': round(pred1, 2),
        'model2_prediction': round(pred2, 2) if pred2 is not None else None,
        'households': int(row.get('households', 0))
    })

# Sort by commercial growth rate (descending)
predictions.sort(key=lambda x: x['commercial_growth_rate'], reverse=True)

# Add ranking
for idx, pred in enumerate(predictions, 1):
    pred['rank'] = idx

# Output
output = {
    'success': True,
    'total_wards': len(predictions),
    'model1_accuracy': {
        'test_r2': metadata['model1']['test_r2'],
        'cv_r2': metadata['model1']['cv_r2_mean']
    },
    'model2_accuracy': {
        'test_r2': metadata['model2']['test_r2'],
        'cv_r2': metadata['model2']['cv_r2_mean']
    } if metadata.get('model2') else None,
    'predictions': predictions
}

print(json.dumps(output, indent=2))

