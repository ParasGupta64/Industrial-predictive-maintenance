# Industrial Predictive Maintenance Platform

## 1. Project Overview
The Industrial Predictive Maintenance Platform is an AI-powered web application designed to predict machine failures before they occur. It uses a Machine Learning model to analyze machine operating conditions and Google's Gemini AI to provide actionable maintenance insights.

## 2. Problem Statement
Industrial machines undergo significant wear and tear, and unexpected failures lead to costly downtime. By predicting failures based on sensor data, operators can perform maintenance proactively rather than reactively.

## 3. Features
- **User Authentication:** Secure login and registration system.
- **Dashboard Overview:** Displays machine health summary and user statistics.
- **Machine Failure Prediction:** Machine learning model to predict failure risk.
- **AI Analysis:** Integration with Gemini AI for detailed, human-readable maintenance recommendations.
- **Prediction History:** Keeps a log of all predictions made by the user.

## 4. Machine Learning Model
The platform uses a pre-trained scikit-learn model (`machine_failure_model.pkl`).
Inputs include: Air Temperature (K), Process Temperature (K), Rotational Speed (rpm), Torque (Nm), Tool Wear (min), and Machine Type (L, M, H).

## 5. Gemini AI Integration
Gemini AI acts as a maintenance assistant. It takes the machine's parameters and the ML prediction, then provides possible contributing factors and recommended actions without overriding the ML model's decision.

## 6. Technology Stack
- **Backend:** Flask, Python, SQLite (Flask-SQLAlchemy)
- **Frontend:** HTML, CSS, JavaScript (Vanilla)
- **Machine Learning:** scikit-learn, joblib
- **AI/LLM:** Google GenAI SDK (Gemini)

## 7. Application Workflow
1. User logs in.
2. User enters machine sensor readings on the Prediction page.
3. The ML model predicts failure risk (0 or 1).
4. Gemini generates an explanation based on the input and ML prediction.
5. The result is saved and displayed to the user.

## 8. Project Structure
```text
MachineFailurePrediction/
├── app.py
├── llm_explaination.py
├── machine_failure_model.pkl
├── requirements.txt
├── .env
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── predict.html
    ├── result.html
    ├── analysis.html
    ├── history.html
    └── profile.html
```

## 9. Installation
1. Clone the repository or download the project files.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 10. Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API Key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## 11. How to Run
Run the Flask application:
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

## 12. User Flow
- **Register / Login** -> Access the **Dashboard**.
- Click **New Prediction** -> Fill in machine data -> View **Result**.
- Click **View Gemini AI Analysis** -> Read AI insights.
- Go to **Analysis History** -> Review past predictions.

## 13. Screenshots
*(Placeholder for screenshots)*

## 14. Limitations
- The application relies on the Gemini API for explanations; network connectivity is required.
- The ML model operates on specific input bounds based on its training data.

## 15. Future Improvements
- Add role-based access control (Admin vs Operator).
- Real-time sensor data streaming integration.
- Export prediction history to CSV or PDF.
