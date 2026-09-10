import pytest
from app import Prediction

def test_invalid_ranges(logged_in_client, session):
    """Test that invalid values within the dataset ranges flash validation errors."""
    initial_count = Prediction.query.count()
    
    # Process temp < Air temp
    response = logged_in_client.post('/predict', data={
        'air_temperature': '300.0',
        'process_temperature': '298.0',
        'rotational_speed': '1500.0',
        'torque': '40.0',
        'tool_wear': '50.0',
        'type': 'L'
    })
    
    assert response.status_code == 200
    assert b"Process Temperature (298.0 K) is outside the model training range" in response.data
    
    # Negative/out of bounds tool wear
    response2 = logged_in_client.post('/predict', data={
        'air_temperature': '300.0',
        'process_temperature': '310.0',
        'rotational_speed': '1500.0',
        'torque': '40.0',
        'tool_wear': '-10.0',
        'type': 'L'
    })
    
    assert b"Tool Wear (-10.0 min) is outside the model training range" in response2.data
    
    # Verify no predictions were saved to the DB
    assert Prediction.query.count() == initial_count

def test_missing_fields_or_invalid_type(logged_in_client, session):
    """Test non-numeric values and invalid types are rejected."""
    response = logged_in_client.post('/predict', data={
        'air_temperature': 'abc',
        'process_temperature': '310.0',
        'rotational_speed': '1500.0',
        'torque': '40.0',
        'tool_wear': '50.0',
        'type': 'Z'
    })
    
    assert b"Invalid numeric values provided" in response.data
    
    response2 = logged_in_client.post('/predict', data={
        'air_temperature': '300.0',
        'process_temperature': '310.0',
        'rotational_speed': '1500.0',
        'torque': '40.0',
        'tool_wear': '50.0',
        'type': 'Z' # Invalid machine type
    })
    
    assert b"Please select a valid machine type" in response2.data
