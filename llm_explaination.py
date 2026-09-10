import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def generate_failure_explanation(
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        machine_type,
        prediction
        ):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        prompt = f"""
You are an AI-powered industrial maintenance assistant.

An ML model has already analyzed the machine data and produced the following prediction:

ML Prediction: {prediction}

The current machine operating conditions are:
- Air Temperature: {air_temperature} K
- Process Temperature: {process_temperature} K
- Rotational Speed: {rotational_speed} rpm
- Torque: {torque} Nm
- Tool Wear: {tool_wear} minutes
- Machine Type: {machine_type}

Your task is to explain the ML model's prediction in simple, professional, and understandable language for a machine operator.

IMPORTANT RULES:
1. DO NOT make a new prediction. The ML prediction provided above is FINAL.
2. DO NOT invent numerical safety limits, failure thresholds, industry standards, tool-life limits, or recommended operating ranges.
3. DO NOT assume that Machine Type (L, M, or H) represents machine quality, strength, capacity, or tolerance. Treat it only as a categorical machine characteristic used by the ML model.
4. DO NOT claim that a specific component has failed. Mention components that could reasonably be inspected as possible areas for inspection.
5. Base your explanation primarily on the provided machine inputs. Do not introduce sensor values or information that was not provided.
6. Do not treat a single sensor value as proof of failure. Explain the relationship between multiple parameters where relevant.
7. If the exact cause of the prediction cannot be determined from the available inputs, clearly state that the ML model predicts the condition but the exact physical cause requires further inspection or additional sensor/history data.
8. Clearly distinguish between:
   - What the provided data shows
   - Possible contributing factors
   - Recommended maintenance actions
9. Avoid statements such as "this value is definitely safe", "this value is definitely dangerous", or "failure will occur at X minutes" unless such a threshold is explicitly provided in the input.
10. Keep the explanation practical and concise. Do not produce an unnecessarily long technical report.

IF ML PREDICTION = 1:
- State clearly that the ML model predicts machine failure risk.
- Identify the provided parameters that may be contributing to the prediction.
- Explain possible thermal, mechanical, or tool-related effects only when supported by the available data.
- Suggest practical inspection and maintenance actions.
- Do NOT claim to know the exact failure component or root cause.

IF ML PREDICTION = 0:
- State clearly that the ML model predicts no machine failure.
- Explain which provided parameters appear relatively stable based on the available information.
- Mention any parameter that deserves continued monitoring.
- Give general preventive maintenance recommendations.
- Do NOT claim that the machine is guaranteed to be safe or failure-free.

Use the following structure:

### Prediction Summary
Briefly explain the ML prediction.

### Key Parameter Analysis
Explain the important machine inputs and how they relate to the prediction.

### Possible Contributing Factors
Explain possible reasons or factors associated with the prediction. If there is insufficient evidence, say so clearly.

### Maintenance Recommendations
Give practical maintenance or monitoring recommendations.

### Important Limitation
Explain what cannot be determined from the available sensor data.

Remember:
The ML model makes the prediction.
You only explain the prediction.
You must not replace or override the ML model's decision.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        raise e
