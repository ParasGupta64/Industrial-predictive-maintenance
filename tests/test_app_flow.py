import pytest
from app import User, Prediction

def test_homepage_redirect(client):
    """Test that homepage redirects to login for unauthenticated users."""
    response = client.get('/')
    assert response.status_code == 302
    assert '/login' in response.headers.get('Location', '')

def test_registration_and_login(client, db, session):
    """Test full registration and login flow."""
    # Register
    response = client.post('/register', data={
        'username': 'newuser',
        'email': 'new@test.com',
        'password': 'password123',
        'confirm_password': 'password123'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b"Registration successful" in response.data
    
    user = User.query.filter_by(email='new@test.com').first()
    assert user is not None
    
    # Duplicate registration
    response2 = client.post('/register', data={
        'username': 'newuser',
        'email': 'new@test.com',
        'password': 'password123',
        'confirm_password': 'password123'
    }, follow_redirects=True)
    assert b"Email or Username already exists" in response2.data
    
    # Login
    response3 = client.post('/login', data={
        'email': 'new@test.com',
        'password': 'password123'
    }, follow_redirects=True)
    
    assert b"Dashboard" in response3.data
    
    # Logout to clean up session
    client.get('/logout', follow_redirects=True)

def test_protected_routes(client, test_user):
    """Test that protected pages require authentication."""
    routes = ['/dashboard', '/predict', '/history', '/profile']
    
    for route in routes:
        response = client.get(route, follow_redirects=True)
        assert b"Please log in to access this page." in response.data or response.request.path.endswith('/login')

def test_valid_prediction_flow(logged_in_client, session, mock_gemini):
    """Test a valid prediction and verify result & history pages."""
    initial_count = Prediction.query.count()
    
    response = logged_in_client.post('/predict', data={
        'air_temperature': '300.0',
        'process_temperature': '310.0',
        'rotational_speed': '1500.0',
        'torque': '40.0',
        'tool_wear': '50.0',
        'type': 'L'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert Prediction.query.count() == initial_count + 1
    
    prediction = session.query(Prediction).order_by(Prediction.id.desc()).first()
    assert prediction is not None
    assert prediction.gemini_explanation == "This is a mocked Gemini explanation."
    
    # History page
    history_response = logged_in_client.get('/history')
    assert history_response.status_code == 200
    assert str(prediction.id).encode() in history_response.data
    
    # Profile page
    profile_response = logged_in_client.get('/profile')
    assert profile_response.status_code == 200
    assert b"Total Predictions" in profile_response.data
    
    # Analysis page
    analysis_response = logged_in_client.get(f'/analysis/{prediction.id}')
    assert analysis_response.status_code == 200
    assert b"This is a mocked Gemini explanation." in analysis_response.data
