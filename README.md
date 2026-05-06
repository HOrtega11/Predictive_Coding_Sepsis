# Predictive Coding for Sepsis Vital-Sign Forecasting

## Team Members
- Horus Ortega
- Sidharth Nayak

---

## Project Description

This project aims to apply a predictive coding (PC) model using a GRU cell that continuously monitors patient ICU data and forecasts the short term future of vital signs. The PC receives the vital signs for a given patient over a previous fixed window of time, and it predicts those vital signs at future time points. The model’s predictions are compared with those of more traditional machine learning models (baseline GRU and baseline ARIMA) to determine its relative predictive accuracy. The goal is to assess whether PC improvements in clinical forecasting.

The models are evaluated on the following hourly vital signs from the MIMIC-IV demo dataset:
- Heart rate  
- Blood pressure (systolic, diastolic)  
- Temperature  
- Respiratory rate  
- Oxygen saturation  


---

## Environment Setup
All experiments were performed using Python 3.12.3
### 1. Clone repository
### 2. Create a virtual environment (venv)
### 3. Install dependencies from requirements.txt
### 4. Dataset
Download the MIMIC-IV demo dataset. It can be found in https://physionet.org/content/mimic-iv-demo/2.2/
### 5a. Quick run script
Runs a self-contained lightweight version of the full modeling pipeline without external downloads or setup required that verifies the pipeline is running correctly. It produces quick_run_results.csv containing PC_GRU_quick and GRU_quick results.
Run in a terminal using:
python quick_run.py
### 5b. Full Pipeline 
1. Preprocess data: python -m src.data.data_preprocess

2. Train PC-GRU using: python -m src.training.train_pc_gru

3. Train GRU baseline using: python -m src.training.train_gru

4. Run ARIMA baseline using: python -m src.training.train_arima_windowed

5. Generate results, tables, and plots using: python -m src.evaluation.results_tables_plots

**Outputs:**

Data: 
- data/processed/splits/train.csv
- data/processed/splits/val.csv
- data/processed/splits/test.csv
- data/processed/splits/scaler.pkl

Metrics: 
- pc_gru_results.csv
- gru_results.csv
- arima_windowed_results.csv

Plots:
- loss curves
- metric comparisons

Tables
- model_summary_table.csv
- best_model_by_metric.csv
- per_variable_mae_table.csv

### 6. Interactive Dashboard
Run using: streamlit run src/dashboard/dashboard.py

---
### Citations
Lotter, W., Kreiman, G., & Cox, D. (2016). Deep predictive coding networks for video prediction and unsupervised learning. arXiv preprint arXiv:1605.08104.

Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dp1f-ex47

---
## Appendix
OpenAI’s ChatGPT was used to check and correct grammar and wording mistakes. Generative was used to generate code as this project focuses on overall design and practical implementation. All generated code was analyzed to verify correct coding.


