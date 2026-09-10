import pytest
import os
import sys
from unittest.mock import patch

# Set TESTING env var before importing app so it uses in-memory DB
os.environ['TESTING'] = 'true'

# Add the project directory to path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app, db as _db, User

@pytest.fixture(scope='session')
def app():
    """Create and configure a new app instance for each test session."""
    # Create a completely independent test configuration
    flask_app.config.update({
        'TESTING': True,
        'LOGIN_DISABLED': False,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test_secret_key'
    })
    
    yield flask_app

@pytest.fixture(scope='session')
def db(app):
    """Create database for the tests."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()

@pytest.fixture(scope='function')
def session(db):
    """Creates a new database session for a test."""
    db.session.begin_nested()
    
    yield db.session
    
    db.session.rollback()

@pytest.fixture(scope='function')
def client(app, session):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture(scope='function')
def test_user(session):
    """Create a test user."""
    user = session.query(User).filter_by(email='test@test.com').first()
    if not user:
        from werkzeug.security import generate_password_hash
        user = User(username='testuser', email='test@test.com', password_hash=generate_password_hash('password123'))
        session.add(user)
        session.commit()
    return user

@pytest.fixture(scope='function')
def mock_gemini():
    """Mock the Gemini API call."""
    with patch('llm_explaination.genai.Client') as mock_client:
        mock_instance = mock_client.return_value
        mock_response = mock_instance.models.generate_content.return_value
        mock_response.text = "This is a mocked Gemini explanation."
        yield mock_instance

@pytest.fixture(scope='function')
def logged_in_client(client):
    """Return a test client that is logged in as a new user."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    email = f"test_{unique_id}@test.com"
    
    client.post('/register', data={
        'username': f"user_{unique_id}",
        'email': email,
        'password': 'password123',
        'confirm_password': 'password123'
    }, follow_redirects=True)
    
    response = client.post('/login', data={
        'email': email,
        'password': 'password123'
    }, follow_redirects=True)
    
    assert b"Dashboard" in response.data
    yield client
    client.get('/logout', follow_redirects=True)
