# False News Detection Dashboard

**Time Frame:** May 2021 – Feb 2022  
**Technologies:** Python, XGBoost, SHAP, Dash, Pandas

## Project Overview
This repository contains a fake news classification system developed using XGBoost with over 25 NLP and metadata features extracted from COVID-era datasets. The system includes SHAP explainability for model transparency and a Dash dashboard for interactive visualization of risk scores.

## Repository Structure
- `data/` – Raw and processed datasets  
- `notebooks/` – Exploratory analysis and model development notebooks  
- `src/` – Model training and preprocessing scripts  
- `app/` – Dash application source code  
- `requirements.txt` – Python dependencies  
- `.gitignore` – Files and directories to ignore  

## Installation
```bash
git clone https://github.com/yourusername/fake-news-dashboard.git
cd fake-news-dashboard
pip install -r requirements.txt
```

## Usage
- To train the model:
  ```bash
  python src/train_model.py
  ```
- To run the dashboard:
  ```bash
  python app/app.py
  ```

## License
MIT License
