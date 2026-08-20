# Fake Job Detection System

A machine-learning web application that predicts whether a job posting is genuine or fraudulent.

## Features

- Upload job-posting data in CSV format
- Predicts genuine or fake job postings
- Shows fraud probability for each posting
- Includes visualizations for prediction results
- Uses pre-trained machine-learning models

## Project Structure

```text
fake-job-detector/
├── app.py
├── data/
│   ├── train.csv
│   └── train_model.py
├── models/
│   ├── fake_job_detector.pkl
│   ├── model.pkl
│   └── vectorizer.pkl
└── README.md

Run the Application
1. Install the required Python packages:
   pip install streamlit pandas scikit-learn plotly
2. Start the app:
   streamlit run app.py
3. Open the local link shown in the terminal and upload a CSV file containing job-posting data.


Technologies Used:
- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly


Author:
Anjaneya Mohapatra