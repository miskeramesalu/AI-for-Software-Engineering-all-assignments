import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic patient records
n_patients = 500

# Generate synthetic hospital data
synthetic_data = pd.DataFrame({
    "age": np.random.randint(20, 90, size=n_patients),
    "gender": np.random.choice(["Male", "Female"], size=n_patients),
    "length_of_stay": np.random.randint(1, 15, size=n_patients),
    "previous_admissions": np.random.randint(0, 5, size=n_patients),
    "num_medications": np.random.randint(1, 20, size=n_patients),
    "readmission_30d": np.random.choice([0, 1], size=n_patients, p=[0.8, 0.2]),
    "comorbidity_score": np.random.randint(0, 10, size=n_patients),
    "insurance_type": np.random.choice(["Private", "Medicare", "Medicaid", "Uninsured"], size=n_patients),
    "distance_to_hospital_km": np.round(np.random.uniform(0.5, 50.0, size=n_patients), 2)
})

# Save to CSV
synthetic_data.to_csv("data/raw/hospital_data.csv", index=False)
print("✅ Synthetic hospital_data.csv created successfully in data/raw/")
