import argparse
import os
import joblib
import pandas as pd

def get_grade_and_feedback(score):
    if score >= 90:
        grade = "A+ (Outstanding)"
        feedback = "Exceptional performance! Maintain your consistent study habits."
    elif score >= 80:
        grade = "A (Excellent)"
        feedback = "Great work! A slight increase in study hours could elevate you to A+."
    elif score >= 70:
        grade = "B (Good)"
        feedback = "Solid performance. Regular revision and higher attendance will push this to an A."
    elif score >= 60:
        grade = "C (Satisfactory)"
        feedback = "Passing grade, but there is room for improvement. Focus on weaker subject areas."
    elif score >= 50:
        grade = "D (Needs Improvement)"
        feedback = "At risk! Increasing daily study time by 2-3 hours and attending all lectures is strongly advised."
    else:
        grade = "F (Fail / Critical Warning)"
        feedback = "Urgent attention required. High risk of failing exam. Seek academic guidance and increase study hours immediately."
    return grade, feedback

def predict_score(study_hours: float, attendance: float, previous_marks: float, model_path="models/student_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'python train.py' first.")
        
    model = joblib.load(model_path)
    
    # Prepare input dataframe
    input_data = pd.DataFrame([{
        "study_hours": float(study_hours),
        "attendance": float(attendance),
        "previous_marks": float(previous_marks)
    }])
    
    predicted_score = model.predict(input_data)[0]
    predicted_score = max(0.0, min(100.0, round(float(predicted_score), 2)))
    
    grade, feedback = get_grade_and_feedback(predicted_score)
    return predicted_score, grade, feedback

def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction CLI")
    parser.add_argument("--hours", type=float, help="Daily study hours (e.g. 5.5)")
    parser.add_argument("--attendance", type=float, help="Attendance percentage (e.g. 85.0)")
    parser.add_argument("--previous", type=float, help="Previous marks out of 100 (e.g. 78.0)")
    
    args = parser.parse_args()
    
    # If arguments not provided via CLI, prompt interactively
    print("==================================================")
    print("      STUDENT PERFORMANCE PREDICTION AI CLI       ")
    print("==================================================")
    
    try:
        hours = args.hours if args.hours is not None else float(input("Enter Daily Study Hours (1.0 - 12.0): "))
        attendance = args.attendance if args.attendance is not None else float(input("Enter Attendance Percentage (0 - 100)%: "))
        previous = args.previous if args.previous is not None else float(input("Enter Previous Exam Marks (0 - 100): "))
    except ValueError:
        print("\n[Error] Please enter valid numeric values for all inputs.")
        return

    # Validations
    if hours < 0 or hours > 24:
        print("\n[Warning] Study hours should typically be between 0 and 16.")
    if attendance < 0 or attendance > 100:
        print("\n[Error] Attendance must be between 0% and 100%.")
        return
    if previous < 0 or previous > 100:
        print("\n[Error] Previous marks must be between 0 and 100.")
        return

    score, grade, feedback = predict_score(hours, attendance, previous)
    
    print("\n---------------- PREDICTION REPORT ----------------")
    print(f" Input Features:")
    print(f"   • Daily Study Hours  : {hours} hrs/day")
    print(f"   • Attendance Rate    : {attendance}%")
    print(f"   • Previous Exam Marks: {previous} / 100")
    print("---------------------------------------------------")
    print(f" Predicted Final Exam Score : {score} / 100")
    print(f" Predicted Grade            : {grade}")
    print(f" Academic Feedback          : {feedback}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    main()
