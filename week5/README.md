# AI Development Workflow Assignment

This repository contains all files related to the AI Development Workflow Assignment for the **AI for Software Engineering** course.

## Contents
- **docs/** → Assignment write-up (PDF/Markdown)
- **src/** → Source code for data preprocessing, feature engineering, training, and serving
- **notebooks/** → Jupyter notebooks for EDA and experimentation
- **models/** → Trained model artifacts
- **outputs/** → Generated logs and charts
- **data/** → Folder for raw and processed datasets (not included in repo for privacy)

## Tools
- Python 3.10+
- scikit-learn
- pandas, numpy
- matplotlib / seaborn
- LightGBM or XGBoost

## Run
```bash
pip install -r requirements.txt
python src/train.py
