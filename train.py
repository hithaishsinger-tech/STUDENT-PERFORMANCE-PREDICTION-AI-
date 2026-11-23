import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_evaluate():
    data_path = "data/student_data.csv"
    if not os.path.exists(data_path):
        from generate_dataset import generate_student_dataset
        print("Dataset not found. Generating default student dataset...")
        generate_student_dataset()
        
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} student records.")
    
    # Feature matrix X and Target vector y
    feature_cols = ["study_hours", "attendance", "previous_marks"]
    target_col = "final_score"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 80% Training, 20% Testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_r2 = -float("inf")
    results = {}
    
    print("\n================ MODEL EVALUATION RESULTS ================")
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (MAPE) and Accuracy %
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy_percentage = max(0.0, 100.0 - mape)
        
        results[name] = {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "accuracy_percentage": round(float(accuracy_percentage), 2)
        }
        
        print(f"\nModel: {name}")
        print(f"  - R² Score (Variance Explained): {r2 * 100:.2f}% (R² = {r2:.4f})")
        print(f"  - Accuracy (100 - MAPE):         {accuracy_percentage:.2f}%")
        print(f"  - Mean Absolute Error (MAE):     {mae:.2f} marks")
        print(f"  - Root Mean Squared Error (RMSE):{rmse:.2f} marks")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print("\n==========================================================")
    print(f"Best Performing Model: {best_model_name} with R² = {best_r2 * 100:.2f}%")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/student_model.pkl"
    joblib.dump(best_model, model_save_path)
    print(f"Trained model saved to: {model_save_path}")
    
    # Save training metrics and metadata
    metrics_path = "models/model_metrics.json"
    metadata = {
        "best_model": best_model_name,
        "feature_names": feature_cols,
        "target": target_col,
        "metrics": results[best_model_name],
        "all_model_results": results
    }
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    return best_model, metadata

if __name__ == "__main__":
    train_and_evaluate()
