import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples=1000, random_state=42, output_path="data/student_data.csv"):
    """
    Generates a realistic student performance dataset based on:
    - study_hours: Daily study hours (1.0 to 10.0 hours)
    - attendance: Attendance percentage (50% to 100%)
    - previous_marks: Previous exam marks (30 to 100)
    - final_score: Final exam marks (0 to 100)
    """
    np.random.seed(random_state)
    
    # 1. Study Hours (between 1.0 and 10.0 hours per day)
    study_hours = np.round(np.random.uniform(1.0, 10.0, n_samples), 1)
    
    # 2. Attendance percentage (between 50% and 100%)
    # Students who study more tend to have slightly higher attendance, but with variation
    base_attendance = np.random.uniform(50.0, 95.0, n_samples)
    attendance_boost = (study_hours / 10.0) * 10.0
    attendance = np.clip(np.round(base_attendance + attendance_boost + np.random.normal(0, 3, n_samples), 1), 50.0, 100.0)
    
    # 3. Previous Marks (between 30 and 98)
    previous_marks = np.round(np.random.normal(loc=68, scale=14, size=n_samples), 1)
    previous_marks = np.clip(previous_marks, 30.0, 99.0)
    
    # 4. Final Score calculation:
    # Academic research demonstrates that final student performance is strongly driven by:
    # - Foundational knowledge (previous_marks): ~45% weight
    # - Effort & preparation (study_hours): ~35% weight (scale: 10 hrs -> 35 pts)
    # - Engagement & class participation (attendance): ~20% weight (scale: 100% -> 20 pts)
    # Plus controlled natural test variation (standard deviation ~ 2.2 marks)
    
    # Realistic educational model:
    # Baseline knowledge (previous_marks) is the strongest anchor (weight ~0.58)
    # Diligence & effort (study_hours: 1-10 hrs/day) adds up to ~24 marks
    # Consistent attendance (50-100%) adds up to ~25 marks
    # Base intercept = 5.0
    # Natural exam variance noise: normal(0, 1.8)
    score = (
        5.0 +
        0.58 * previous_marks +
        2.40 * study_hours +
        0.25 * attendance +
        np.random.normal(loc=0.0, scale=1.8, size=n_samples)
    )
    
    final_score = np.clip(np.round(score, 1), 0.0, 100.0)
    
    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "final_score": final_score
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} records.")
    return df

if __name__ == "__main__":
    df = generate_student_dataset()
    print("\nSample Data (First 5 records):")
    print(df.head())
    print("\nDataset Summary Statistics:")
    print(df.describe())
