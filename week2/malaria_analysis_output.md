malaria_analysis.ipynb output

🔬 Malaria Outbreak Prediction Analysis
==================================================
Dataset shape: (2710, 14)
Columns: ['IND_ID', 'IND_CODE', 'IND_UUID', 'IND_PER_CODE', 'DIM_TIME', 'DIM_TIME_TYPE', 'DIM_GEO_CODE_M49', 'DIM_GEO_CODE_TYPE', 'DIM_PUBLISH_STATE_CODE', 'IND_NAME', 'GEO_NAME_SHORT', 'RATE_PER_1000_N', 'RATE_PER_1000_NL', 'RATE_PER_1000_NU']

📊 Dataset Overview:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2710 entries, 0 to 2709
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   IND_ID                  2710 non-null   object 
 1   IND_CODE                2710 non-null   object 
 2   IND_UUID                2710 non-null   object 
 3   IND_PER_CODE            2710 non-null   object 
 4   DIM_TIME                2710 non-null   int64  
 5   DIM_TIME_TYPE           2710 non-null   object 
 6   DIM_GEO_CODE_M49        2710 non-null   int64  
 7   DIM_GEO_CODE_TYPE       2710 non-null   object 
 8   DIM_PUBLISH_STATE_CODE  2710 non-null   object 
 9   IND_NAME                2710 non-null   object 
 10  GEO_NAME_SHORT          2710 non-null   object 
 11  RATE_PER_1000_N         0 non-null      float64
 12  RATE_PER_1000_NL        0 non-null      float64
 13  RATE_PER_1000_NU        0 non-null      float64
dtypes: float64(3), int64(2), object(9)
memory usage: 296.5+ KB
None

🔍 Missing Values Analysis:
IND_ID                       0
IND_CODE                     0
IND_UUID                     0
IND_PER_CODE                 0
DIM_TIME                     0
DIM_TIME_TYPE                0
DIM_GEO_CODE_M49             0
DIM_GEO_CODE_TYPE            0
DIM_PUBLISH_STATE_CODE       0
IND_NAME                     0
GEO_NAME_SHORT               0
RATE_PER_1000_N           2710
RATE_PER_1000_NL          2710
RATE_PER_1000_NU          2710
dtype: int64

🌍 Unique Countries: 115
📅 Year Range: 2000 - 2023

⚙️  Data Preprocessing...

📈 Sample of simulated data:
   DIM_TIME  DIM_GEO_CODE_M49                         GEO_NAME_SHORT  \
0      2001               364             Iran (Islamic Republic of)   
1      2001               368                                   Iraq   
2      2001               384                          Côte d'Ivoire   
3      2001               404                                  Kenya   
4      2001               408  Democratic People's Republic of Korea   
5      2001               410                      Republic of Korea   
6      2001               417                             Kyrgyzstan   
7      2001               418       Lao People's Democratic Republic   
8      2001               430                                Liberia   
9      2001               450                             Madagascar   

  DIM_GEO_CODE_TYPE  incidence_rate  
0           COUNTRY             0.0  
1           COUNTRY             0.0  
2           COUNTRY             0.0  
3           COUNTRY             0.0  
4           COUNTRY             0.0  
5           COUNTRY             0.0  
6           COUNTRY             0.0  
7           COUNTRY             0.0  
8           COUNTRY             0.0  
9           COUNTRY             0.0  

⚙️  Feature Engineering...
✅ Feature engineering completed

📊 E🔬 Malaria Outbreak Prediction Analysis
==================================================
Dataset shape: (2710, 14)
Columns: ['IND_ID', 'IND_CODE', 'IND_UUID', 'IND_PER_CODE', 'DIM_TIME', 'DIM_TIME_TYPE', 'DIM_GEO_CODE_M49', 'DIM_GEO_CODE_TYPE', 'DIM_PUBLISH_STATE_CODE', 'IND_NAME', 'GEO_NAME_SHORT', 'RATE_PER_1000_N', 'RATE_PER_1000_NL', 'RATE_PER_1000_NU']

📊 Dataset Overview:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2710 entries, 0 to 2709
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   IND_ID                  2710 non-null   object 
 1   IND_CODE                2710 non-null   object 
 2   IND_UUID                2710 non-null   object 
 3   IND_PER_CODE            2710 non-null   object 
 4   DIM_TIME                2710 non-null   int64  
 5   DIM_TIME_TYPE           2710 non-null   object 
 6   DIM_GEO_CODE_M49        2710 non-null   int64  
 7   DIM_GEO_CODE_TYPE       2710 non-null   object 
 8   DIM_PUBLISH_STATE_CODE  2710 non-null   object 
 9   IND_NAME                2710 non-null   object 
 10  GEO_NAME_SHORT          2710 non-null   object 
 11  RATE_PER_1000_N         0 non-null      float64
 12  RATE_PER_1000_NL        0 non-null      float64
 13  RATE_PER_1000_NU        0 non-null      float64
dtypes: float64(3), int64(2), object(9)
memory usage: 296.5+ KB
None

🔍 Missing Values Analysis:
IND_ID                       0
IND_CODE                     0
IND_UUID                     0
IND_PER_CODE                 0
DIM_TIME                     0
DIM_TIME_TYPE                0
DIM_GEO_CODE_M49             0
DIM_GEO_CODE_TYPE            0
DIM_PUBLISH_STATE_CODE       0
IND_NAME                     0
GEO_NAME_SHORT               0
RATE_PER_1000_N           2710
RATE_PER_1000_NL          2710
RATE_PER_1000_NU          2710
dtype: int64

🌍 Unique Countries: 115
📅 Year Range: 2000 - 2023

⚙️  Data Preprocessing...

📈 Sample of simulated data:
   DIM_TIME  DIM_GEO_CODE_M49                         GEO_NAME_SHORT  \
0      2001               364             Iran (Islamic Republic of)   
1      2001               368                                   Iraq   
2      2001               384                          Côte d'Ivoire   
3      2001               404                                  Kenya   
4      2001               408  Democratic People's Republic of Korea   
5      2001               410                      Republic of Korea   
6      2001               417                             Kyrgyzstan   
7      2001               418       Lao People's Democratic Republic   
8      2001               430                                Liberia   
9      2001               450                             Madagascar   

  DIM_GEO_CODE_TYPE  incidence_rate  
0           COUNTRY             0.0  
1           COUNTRY             0.0  
2           COUNTRY             0.0  
3           COUNTRY             0.0  
4           COUNTRY             0.0  
5           COUNTRY             0.0  
6           COUNTRY             0.0  
7           COUNTRY             0.0  
8           COUNTRY             0.0  
9           COUNTRY             0.0  

⚙️  Feature Engineering...
✅ Feature engineering completed

📊 Exploratory Data Analysisxploratory Data Analysis

![alt text](image.png)
![alt text](image-1.png)
🤖 Machine Learning Model Development
Training set: 2047 samples
Test set: 663 samples

Training Random Forest Regressor...

📈 Model Evaluation Results:
========================================
Mean Absolute Error (MAE): 8.01
Root Mean Squared Error (RMSE): 11.75
R² Score: 0.79

🔍 Feature Importance:
               feature  importance
0  prev_year_incidence    0.936194
1           time_index    0.036383
2          region_code    0.027422
![alt text](image-2.png)
💾 Model saved as 'malaria_model.pkl'

🚨 Outbreak Risk Assessment
Outbreak threshold: 2.83
High-risk regions: ['Afghanistan', 'Africa', 'Algeria', 'Americas', 'Angola', 'Argentina', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Belize', 'Benin', 'Bhutan', 'Bolivia (Plurinational State of)', 'Botswana', 'Brazil', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Central African Republic', 'Chad', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', "Côte d'Ivoire", "Democratic People's Republic of Korea", 'Democratic Republic of the Congo', 'Djibouti', 'Dominican Republic', 'Eastern Mediterranean', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Europe', 'French Guiana', 'Gabon', 'Gambia', 'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mexico', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Republic of Korea', 'Rwanda', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'South-East Asia', 'Sri Lanka', 'Sudan', 'Suriname', 'Syrian Arab Republic', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Togo', 'Turkmenistan', 'Türkiye', 'Uganda', 'United Arab Emirates', 'United Republic of Tanzania', 'Uzbekistan', 'Vanuatu', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Western Pacific', 'World', 'Yemen', 'Zambia', 'Zimbabwe']

==================================================
✅ Analysis Complete!
- Model R² score: 0.79
- Most important feature: prev_year_incidence
- High-risk regions: 113
==================================================

malaria_forecast.py out put

MALARIA OUTBREAK PREDICTION SYSTEM
==================================================
SDG 3: Good Health and Well-being
Python version: 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]
Current directory: e:\PLPAcademy\AI For Software Engineering Specilization\Week2\week2-assignment
==================================================

[DATA] Loading current data...
SUCCESS: Current data loaded

CURRENT MALARIA SITUATION:
==================================================
-> Africa                    | Incidence:   89.2
-> Americas                  | Incidence:   39.2
-> South-East Asia           | Incidence:   83.8
-> Europe                    | Incidence:   31.4
-> Eastern Mediterranean     | Incidence:   18.3
-> Western Pacific           | Incidence:   20.2

[AI] Generating predictions...
Using simulated predictions (demo mode)

PREDICTION RESULTS:
======================================================================
[MED]   Africa                    | Current:   89.2 | Predicted:  104.8 | Risk: Medium
[LOW]   Americas                  | Current:   39.2 | Predicted:   42.8 | Risk: Low
[LOW]   South-East Asia           | Current:   83.8 | Predicted:   96.0 | Risk: Low
[LOW]   Europe                    | Current:   31.4 | Predicted:   28.9 | Risk: Low
[LOW]   Eastern Mediterranean     | Current:   18.3 | Predicted:   15.2 | Risk: Low
[LOW]   Western Pacific           | Current:   20.2 | Predicted:   16.8 | Risk: Low

OUTBREAK ALERTS:
========================================
No critical alerts at this time

SUMMARY STATISTICS:
==============================
Total regions analyzed: 6
High risk regions: 0 (0.0%)
Medium risk regions: 1 (16.7%)
Low risk regions: 5 (83.3%)
Average predicted incidence: 50.7

RECOMMENDATIONS:
==============================
* Maintain current surveillance levels
* Continue preventive measures
* Monitor for seasonal changes

Simulation completed successfully!
Predictions saved to: malaria_predictions.csv