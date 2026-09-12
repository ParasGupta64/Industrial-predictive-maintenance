import os
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import markdown
import bleach
import pandas as pd

from llm_explaination import generate_failure_explanation

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_here')
if os.environ.get('TESTING') == 'true':
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'error'

# Load the model
try:
    loaded_model = joblib.load("machine_failure_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# --- Database Models ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    air_temp = db.Column(db.Float, nullable=False)
    process_temp = db.Column(db.Float, nullable=False)
    rotational_speed = db.Column(db.Float, nullable=False)
    torque = db.Column(db.Float, nullable=False)
    tool_wear = db.Column(db.Float, nullable=False)
    machine_type = db.Column(db.String(1), nullable=False)
    
    ml_prediction = db.Column(db.Integer, nullable=False)
    gemini_explanation = db.Column(db.Text, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Authentication Routes ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email').lower() if request.form.get('email') else ''
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email').lower() if request.form.get('email') else ''
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
            
        user_exists = User.query.filter((User.email == email) | (User.username == username)).first()
        if user_exists:
            flash('Email or Username already exists.', 'error')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Application Routes ---

@app.route('/dashboard')
@login_required
def dashboard():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    healthy_predictions = Prediction.query.filter_by(user_id=current_user.id, ml_prediction=0).count()
    failure_predictions = Prediction.query.filter_by(user_id=current_user.id, ml_prediction=1).count()
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                           total=total_predictions, 
                           healthy=healthy_predictions, 
                           failure=failure_predictions,
                           recent=recent_predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
        
    # POST Request - Process prediction
    try:
        air_temperature = float(request.form.get("air_temperature"))
        process_temperature = float(request.form.get("process_temperature"))
        rotational_speed = float(request.form.get("rotational_speed"))  
        torque = float(request.form.get("torque"))
        tool_wear = float(request.form.get("tool_wear"))
        machine_type = request.form.get("type")
        
        # Strict Validation based on AI4I 2020 dataset ranges
        if not (295.0 <= air_temperature <= 305.0):
            flash(f'Air Temperature ({air_temperature} K) is outside the model training range (295 - 305 K).', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if not (305.0 <= process_temperature <= 315.0):
            flash(f'Process Temperature ({process_temperature} K) is outside the model training range (305 - 315 K).', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if process_temperature < air_temperature:
            flash('Process Temperature cannot be lower than Air Temperature.', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if not (1100 <= rotational_speed <= 2900):
            flash(f'Rotational Speed ({rotational_speed} rpm) is outside the model training range (1100 - 2900 rpm).', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if not (0 <= torque <= 80):
            flash(f'Torque ({torque} Nm) is outside the model training range (0 - 80 Nm).', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if not (0 <= tool_wear <= 260):
            flash(f'Tool Wear ({tool_wear} min) is outside the model training range (0 - 260 min).', 'error')
            return render_template('predict.html', request_form=request.form)
            
        if machine_type not in ["L", "M", "H"]:
            flash('Please select a valid machine type.', 'error')
            return render_template('predict.html', request_form=request.form)
            
        L = 1 if machine_type == "L" else 0
        M = 1 if machine_type == "M" else 0
        H = 1 if machine_type == "H" else 0

        feature_names = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'H', 'L', 'M']
        input_data = pd.DataFrame([[air_temperature, process_temperature, rotational_speed, torque, tool_wear, H, L, M]], columns=feature_names)
        
        if loaded_model is None:
            flash('Machine Learning model is not available.', 'error')
            return redirect(url_for('dashboard'))
            
        prediction = loaded_model.predict(input_data)
        prediction_value = int(prediction[0])
        
        # Get AI explanation
        try:
            explanation = generate_failure_explanation(
                air_temperature, process_temperature, rotational_speed,
                torque, tool_wear, machine_type, prediction_value
            )
        except Exception as e:
            import traceback
            print(f"Gemini API Error: {e}")
            traceback.print_exc()
            explanation = "AI analysis is temporarily unavailable. The machine prediction was completed successfully."
            
        # Save to database
        new_prediction = Prediction(
            user_id=current_user.id,
            air_temp=air_temperature,
            process_temp=process_temperature,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            machine_type=machine_type,
            ml_prediction=prediction_value,
            gemini_explanation=explanation
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        return redirect(url_for('result', prediction_id=new_prediction.id))
        
    except ValueError:
        flash('Invalid numeric values provided.', 'error')
        return render_template('predict.html', request_form=request.form)
    except Exception as e:
        print(f"Prediction Error: {e}")
        flash('An unexpected error occurred during prediction.', 'error')
        return render_template('predict.html', request_form=request.form)

@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    prediction = db.get_or_404(Prediction, prediction_id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('dashboard'))
        
    return render_template('result.html', prediction=prediction)

@app.route('/analysis/<int:prediction_id>')
@login_required
def analysis(prediction_id):
    prediction = db.get_or_404(Prediction, prediction_id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('dashboard'))
        
    raw_markdown = prediction.gemini_explanation
    if not raw_markdown or "temporarily unavailable" in raw_markdown:
        try:
            raw_markdown = generate_failure_explanation(
                prediction.air_temp,
                prediction.process_temp,
                prediction.rotational_speed,
                prediction.torque,
                prediction.tool_wear,
                prediction.machine_type,
                prediction.ml_prediction
            )
            prediction.gemini_explanation = raw_markdown
            db.session.commit()
        except Exception as e:
            import traceback
            print(f"Gemini API Error (Regeneration): {e}")
            traceback.print_exc()
            if not raw_markdown:
                raw_markdown = "AI analysis is unavailable for this prediction."
        
    html_content = markdown.markdown(raw_markdown)
    
    # Safely render HTML
    allowed_tags = list(bleach.ALLOWED_TAGS) + [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'ul', 'ol', 'li', 'strong', 'em', 'br', 'hr'
]
    safe_html = bleach.clean(html_content, tags=allowed_tags, strip=True)
    
    return render_template('analysis.html', prediction=prediction, explanation=safe_html)

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/profile')
@login_required
def profile():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', total_predictions=total_predictions)


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=False)
