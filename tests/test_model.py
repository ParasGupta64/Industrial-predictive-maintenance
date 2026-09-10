import os
import joblib
import pandas as pd

def test_model_exists_and_loads():
    """Test that the machine failure model exists and loads successfully."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'machine_failure_model.pkl')
    assert os.path.exists(model_path), "Model file does not exist."
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        assert False, f"Failed to load model: {e}"
        
    assert model is not None

def test_model_feature_names_and_prediction():
    """Test that the model has the correct feature names and can predict."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'machine_failure_model.pkl')
    model = joblib.load(model_path)
    
    expected_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'H', 'L', 'M']
    
    assert hasattr(model, 'feature_names_in_'), "Model does not have feature_names_in_ attribute."
    assert list(model.feature_names_in_) == expected_features, f"Feature names mismatch. Expected {expected_features}, got {list(model.feature_names_in_)}"
    
    # Test valid prediction
    input_data = pd.DataFrame([[300.0, 310.0, 1500.0, 40.0, 50.0, 0, 1, 0]], columns=expected_features)
    prediction = model.predict(input_data)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1], "Prediction should be binary classification (0 or 1)."
