"""
Model Training Script
---------------------
Trains a LightGBM classifier and saves the model.
"""

import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report
from data_processing import load_data, preprocess_data, split_data

if __name__ == "__main__":
    df = load_data("../data/raw/hospital_data.csv")
    df = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss")

    y_pred = model.predict(X_test)
    print("📊 Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "../models/readmission_model.pkl")
    print("✅ Model saved to models/readmission_model.pkl")
