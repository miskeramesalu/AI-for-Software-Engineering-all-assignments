# malaria_forecast.py
"""
Malaria Outbreak Forecasting System - Windows Compatible Version
SDG 3: Good Health and Well-being
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class MalariaForecaster:
    """Malaria outbreak prediction system"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.features = ['prev_year_incidence', 'time_index', 'region_code']
    
    def predict_outbreak_risk(self, data, threshold=100):
        """Predict malaria outbreak risk with fallback to simulation"""
        print("Using simulated predictions (demo mode)")
        
        # Create realistic predictions based on region patterns
        np.random.seed(42)
        predictions = []
        
        for _, row in data.iterrows():
            region = row['GEO_NAME_SHORT']
            current_rate = row['incidence_rate']
            
            # Region-specific prediction patterns
            if region == 'Africa':
                predicted = current_rate * np.random.uniform(1.1, 1.3)
            elif region == 'South-East Asia':
                predicted = current_rate * np.random.uniform(1.0, 1.2)
            elif region == 'Americas':
                predicted = current_rate * np.random.uniform(0.9, 1.1)
            else:
                predicted = current_rate * np.random.uniform(0.8, 1.0)
            
            # Determine risk level
            if predicted > threshold * 1.2:
                risk_level = 'High'
            elif predicted > threshold:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
                
            predictions.append({
                'region': region,
                'current_incidence': current_rate,
                'predicted_incidence': predicted,
                'risk_level': risk_level
            })
        
        return pd.DataFrame(predictions)

def simulate_current_data():
    """Simulate current malaria data for demonstration"""
    regions = ['Africa', 'Americas', 'South-East Asia', 'Europe', 
               'Eastern Mediterranean', 'Western Pacific']
    
    data = []
    for region in regions:
        # Simulate realistic incidence rates
        if region == 'Africa':
            base_rate = max(5, np.random.normal(120, 20))
        elif region == 'South-East Asia':
            base_rate = max(5, np.random.normal(80, 15))
        elif region == 'Americas':
            base_rate = max(5, np.random.normal(40, 10))
        else:
            base_rate = max(5, np.random.normal(20, 8))
        
        data.append({
            'DIM_TIME': 2020,
            'GEO_NAME_SHORT': region,
            'incidence_rate': base_rate
        })
    
    return pd.DataFrame(data)

def main():
    """Main execution function"""
    print("MALARIA OUTBREAK PREDICTION SYSTEM")
    print("=" * 50)
    print("SDG 3: Good Health and Well-being")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print("=" * 50)
    
    try:
        # Initialize forecaster
        forecaster = MalariaForecaster()
        
        # Load/simulate current data
        print("\n[DATA] Loading current data...")
        current_data = simulate_current_data()
        print("SUCCESS: Current data loaded")
        
        # Display current situation
        print("\nCURRENT MALARIA SITUATION:")
        print("=" * 50)
        for _, row in current_data.iterrows():
            print(f"-> {row['GEO_NAME_SHORT']:25} | Incidence: {row['incidence_rate']:6.1f}")
        
        # Make predictions
        print("\n[AI] Generating predictions...")
        predictions = forecaster.predict_outbreak_risk(current_data, threshold=100)
        
        # Display results
        print("\nPREDICTION RESULTS:")
        print("=" * 70)
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for _, row in predictions.iterrows():
            risk_indicator = "[HIGH] " if row['risk_level'] == 'High' else "[MED]  " if row['risk_level'] == 'Medium' else "[LOW]  "
            print(f"{risk_indicator} {row['region']:25} | "
                  f"Current: {row['current_incidence']:6.1f} | "
                  f"Predicted: {row['predicted_incidence']:6.1f} | "
                  f"Risk: {row['risk_level']}")
            
            # Count risk levels
            if row['risk_level'] == 'High':
                high_risk_count += 1
            elif row['risk_level'] == 'Medium':
                medium_risk_count += 1
            else:
                low_risk_count += 1
        
        # Generate alerts
        print("\nOUTBREAK ALERTS:")
        print("=" * 40)
        high_risk_regions = predictions[predictions['risk_level'] == 'High']
        
        if not high_risk_regions.empty:
            for _, region in high_risk_regions.iterrows():
                alert_level = "CRITICAL" if region['predicted_incidence'] > 150 else "HIGH"
                print(f"*** {alert_level} ALERT: High malaria outbreak risk predicted for {region['region']}")
                print(f"    Predicted Incidence: {region['predicted_incidence']:.1f}")
        else:
            print("No critical alerts at this time")
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print("=" * 30)
        total_regions = len(predictions)
        print(f"Total regions analyzed: {total_regions}")
        print(f"High risk regions: {high_risk_count} ({high_risk_count/total_regions*100:.1f}%)")
        print(f"Medium risk regions: {medium_risk_count} ({medium_risk_count/total_regions*100:.1f}%)")
        print(f"Low risk regions: {low_risk_count} ({low_risk_count/total_regions*100:.1f}%)")
        print(f"Average predicted incidence: {predictions['predicted_incidence'].mean():.1f}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("=" * 30)
        if high_risk_count > 0:
            print("* Increase surveillance in high-risk regions")
            print("* Pre-position medical supplies and treatments")
            print("* Launch public health awareness campaigns")
            print("* Implement vector control measures")
        else:
            print("* Maintain current surveillance levels")
            print("* Continue preventive measures")
            print("* Monitor for seasonal changes")
        
        print(f"\nSimulation completed successfully!")
        
        # Save results to CSV
        try:
            output_file = "malaria_predictions.csv"
            predictions.to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")
        except Exception as e:
            print(f"Note: Could not save predictions to file: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your Python installation and try again.")

if __name__ == "__main__":
    main()