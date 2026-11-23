import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load Model & Metrics
MODEL_PATH = "models/student_model.pkl"
METRICS_PATH = "models/model_metrics.json"

if not os.path.exists(MODEL_PATH):
    from train import train_and_evaluate
    train_and_evaluate()

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Student Performance Prediction API",
    description="Academic analytics engine predicting student exam outcomes using Machine Learning.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Daily study hours")
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    previous_marks: float = Field(..., ge=0.0, le=100.0, description="Previous exam marks")

def grade_rubric(score: float):
    if score >= 90:
        return {
            "grade": "A+",
            "tier": "Outstanding Academic Standing",
            "status": "success",
            "feedback": "Consistent high mastery across foundational curriculum and active engagement.",
            "action_items": [
                "Maintain structured daily review habits.",
                "Explore peer tutoring or honors-level enrichment topics."
            ]
        }
    elif score >= 80:
        return {
            "grade": "A",
            "tier": "Distinction Standing",
            "status": "success",
            "feedback": "Strong performance trajectory with capability of reaching top percentile.",
            "action_items": [
                "Dedicate 45 additional minutes daily to challenging problem sets.",
                "Keep attendance above 90% leading into final revision."
            ]
        }
    elif score >= 70:
        return {
            "grade": "B",
            "tier": "Good Academic Standing",
            "status": "info",
            "feedback": "Solid conceptual retention; performance can be elevated with targeted revision.",
            "action_items": [
                "Increase daily focused study to at least 5.5 hours.",
                "Target weaker chapter topics identified in previous examinations."
            ]
        }
    elif score >= 60:
        return {
            "grade": "C",
            "tier": "Satisfactory / Moderate Risk",
            "status": "warning",
            "feedback": "Meeting passing criteria but vulnerable to difficult exam variants.",
            "action_items": [
                "Attend professor or TA office hours weekly.",
                "Establish a dedicated 2-hour evening study block without distractions.",
                "Raise attendance to at least 85%."
            ]
        }
    elif score >= 50:
        return {
            "grade": "D",
            "tier": "High Academic Risk",
            "status": "danger",
            "feedback": "Borderline passing threshold; high probability of deficit in key subject modules.",
            "action_items": [
                "Schedule immediate academic advising session.",
                "Commit to minimum 3 hours of supervised or structured study daily.",
                "Strict adherence to 90%+ classroom attendance."
            ]
        }
    else:
        return {
            "grade": "F",
            "tier": "Critical Intervention Required",
            "status": "critical",
            "feedback": "Severe performance deficit detected. Immediate academic intervention required.",
            "action_items": [
                "Mandatory academic counseling consultation.",
                "Comprehensive remedial review plan for fundamental concepts.",
                "Remediation of missed classes and coursework assignments."
            ]
        }

@app.post("/api/predict")
def predict_score(data: StudentInput):
    df = pd.DataFrame([{
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "previous_marks": data.previous_marks
    }])
    
    raw_score = float(model.predict(df)[0])
    final_score = max(0.0, min(100.0, round(raw_score, 1)))
    rubric = grade_rubric(final_score)
    
    # Explainable AI: Feature contributions
    intercept = getattr(model, "intercept_", 5.39)
    coefs = getattr(model, "coef_", [2.385, 0.248, 0.576])
    
    c_hours = round(float(coefs[0] * data.study_hours), 2)
    c_att = round(float(coefs[1] * data.attendance), 2)
    c_prev = round(float(coefs[2] * data.previous_marks), 2)
    c_base = round(float(intercept), 2)
    
    # What-if simulations
    df_plus_1h = pd.DataFrame([{"study_hours": min(14.0, data.study_hours + 1.0), "attendance": data.attendance, "previous_marks": data.previous_marks}])
    df_plus_2h = pd.DataFrame([{"study_hours": min(14.0, data.study_hours + 2.0), "attendance": data.attendance, "previous_marks": data.previous_marks}])
    df_plus_att = pd.DataFrame([{"study_hours": data.study_hours, "attendance": min(100.0, data.attendance + 10.0), "previous_marks": data.previous_marks}])
    
    score_plus_1h = max(0.0, min(100.0, round(float(model.predict(df_plus_1h)[0]), 1)))
    score_plus_2h = max(0.0, min(100.0, round(float(model.predict(df_plus_2h)[0]), 1)))
    score_plus_att = max(0.0, min(100.0, round(float(model.predict(df_plus_att)[0]), 1)))
    
    # Target Goal Feasibility
    # final_score = c_base + c0 * hours + c1 * att + c2 * prev
    # required hours for target = (target - c_base - c1*att - c2*prev) / c0
    target_thresholds = {"A+ (90+)": 90.0, "A (80+)": 80.0, "B (70+)": 70.0}
    goal_plans = {}
    for t_name, t_val in target_thresholds.items():
        if final_score >= t_val:
            goal_plans[t_name] = {"achieved": True, "detail": "Already achieving or exceeding target."}
        else:
            needed_hours = (t_val - c_base - c_att - c_prev) / coefs[0]
            needed_att = (t_val - c_base - c_hours - c_prev) / coefs[1]
            goal_plans[t_name] = {
                "achieved": False,
                "needed_hours_daily": round(max(0.5, needed_hours), 1) if needed_hours <= 14.0 else None,
                "needed_attendance_pct": round(min(100.0, max(0.0, needed_att)), 1) if needed_att <= 100.0 else None
            }
            
    # Sensitivity curve: Study hours from 1.0 to 10.0
    curve_points = []
    for h in [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]:
        test_df = pd.DataFrame([{"study_hours": h, "attendance": data.attendance, "previous_marks": data.previous_marks}])
        s = max(0.0, min(100.0, round(float(model.predict(test_df)[0]), 1)))
        curve_points.append({"hours": h, "predicted_score": s})
        
    return {
        "input": {
            "study_hours": data.study_hours,
            "attendance": data.attendance,
            "previous_marks": data.previous_marks
        },
        "predicted_score": final_score,
        "grade": rubric["grade"],
        "tier": rubric["tier"],
        "status": rubric["status"],
        "feedback": rubric["feedback"],
        "action_items": rubric["action_items"],
        "contributions": {
            "baseline": c_base,
            "study_hours": c_hours,
            "attendance": c_att,
            "previous_marks": c_prev
        },
        "what_if": {
            "score_plus_1h": score_plus_1h,
            "delta_1h": round(score_plus_1h - final_score, 1),
            "score_plus_2h": score_plus_2h,
            "delta_2h": round(score_plus_2h - final_score, 1),
            "score_plus_att": score_plus_att,
            "delta_att": round(score_plus_att - final_score, 1)
        },
        "goal_plans": goal_plans,
        "sensitivity_curve": curve_points
    }

@app.get("/api/presets")
def get_presets():
    return [
        {
            "id": "honors",
            "name": "Aarav Sharma",
            "tag": "Honors Candidate",
            "study_hours": 7.5,
            "attendance": 94.0,
            "previous_marks": 88.0,
            "description": "Consistent diligence, solid grasp of curriculum, high attendance."
        },
        {
            "id": "rebound",
            "name": "Maya Lin",
            "tag": "Midterm Rebound",
            "study_hours": 7.0,
            "attendance": 90.0,
            "previous_marks": 64.0,
            "description": "Overcoming lower midterms through rigorous daily practice and high attendance."
        },
        {
            "id": "att_gap",
            "name": "Liam Connor",
            "tag": "Attendance Deficit",
            "study_hours": 5.5,
            "attendance": 62.0,
            "previous_marks": 78.0,
            "description": "Good subject aptitude but compromised by missed classroom lectures."
        },
        {
            "id": "at_risk",
            "name": "Dev Patel",
            "tag": "At-Risk Support",
            "study_hours": 2.0,
            "attendance": 66.0,
            "previous_marks": 48.0,
            "description": "Struggling foundation and low engagement; requires structured tutoring."
        }
    ]

@app.get("/api/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"message": "Metrics unavailable"}

# Serve static files from 'frontend' directory
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Student Analytics Portal on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
