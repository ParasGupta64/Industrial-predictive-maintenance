import sys
import os
sys.path.append(r'C:\Users\PARAS\OneDrive\Desktop\Industrial Predictive Maintenance Platform using Machine Learning\MachineFailurePrediction')
from app import app, db, Prediction
from llm_explaination import generate_failure_explanation

with app.app_context():
    predictions = Prediction.query.all()
    updated = 0
    for pred in predictions:
        if "temporarily unavailable" in pred.gemini_explanation or "unavailable" in pred.gemini_explanation:
            print(f"Fixing Prediction #{pred.id}...")
            try:
                new_explanation = generate_failure_explanation(
                    pred.air_temp,
                    pred.process_temp,
                    pred.rotational_speed,
                    pred.torque,
                    pred.tool_wear,
                    pred.machine_type,
                    pred.ml_prediction
                )
                pred.gemini_explanation = new_explanation
                updated += 1
                print(f"Successfully generated new explanation for Prediction #{pred.id}")
            except Exception as e:
                import traceback
                print(f"Failed to update prediction {pred.id}: {e}")
                traceback.print_exc()
                
    if updated > 0:
        db.session.commit()
        print(f"Fixed {updated} predictions in the database.")
    else:
        print("No predictions needed fixing.")
