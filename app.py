import joblib

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import classification_report, f1_score

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def preprocess(df):
    required_columns = ['title', 'description']

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    text = df['title'].fillna('') + ' ' + df['description'].fillna('')
    return vectorizer.transform(text)

def predict_jobs(df):
    df = df.copy()
    X = preprocess(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df['Predicted Label'] = pd.Series(preds).map({0: 'Real', 1: 'Fake'})
    df['Fraud Probability'] = probs

    return df

def display_prediction_results(df):
    st.subheader("🔍 Predictions")
    st.dataframe(df[["title", "Predicted Label", "Fraud Probability"]])

    st.subheader("🚨 Top 10 Most Suspicious Listings")
    st.table(
        df.sort_values(
            "Fraud Probability",
            ascending=False
        )[["title", "Fraud Probability"]].head(10)
    )

st.set_page_config(page_title="Fake Job Detection System", layout="wide")
st.title("🕵️‍♂️ Fake Job Detection System")
st.markdown(
    "Upload a CSV file containing job listings to identify potentially fraudulent job postings."
)



# === 2. TESTING PHASE ===
st.header("📦 Upload Test Data & View Predictions")
test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test")

if test_file and model and vectorizer:
    
    df_test = pd.read_csv(test_file)
    st.subheader("Test Data Preview")
    st.dataframe(df_test.head())

    # Predict using the trained model
    df_test = predict_jobs(df_test)

    # Keep the original column names used by the UI
    df_test["Predicted Label"] = df_test["Predicted Label"].map({"Real": 0, "Fake": 1})
    df_test["Fraud Probability"] = df_test["Fraud Probability"]
    st.subheader("📊 Fraud Probability Histogram")

    hist = px.histogram(
    df_test,
    x="Fraud Probability",
    nbins=30
    )

    st.plotly_chart(hist, use_container_width=True)


    st.subheader("📊 Predicted Distribution")

    label_counts = df_test["Predicted Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    pie_chart = px.pie(
    label_counts,
    names="Label",
    values="Count",
    title="Real vs Fake Job Distribution"
    )

    st.plotly_chart(pie_chart, use_container_width=True)

    # If true labels are present, show F1
    if "fraudulent" in df_test.columns:
        test_f1 = f1_score(df_test["fraudulent"], df_test["Predicted Label"])
        st.metric("Test F1 Score", f"{test_f1:.4f}")
        st.text("Classification Report")
        st.text(classification_report(df_test["fraudulent"], df_test["Predicted Label"]))
    


elif test_file and not model:
    st.warning("⚠️ Please train the model first before uploading test data.")


st.markdown("---")
st.header("🔍 Check a Single Job Posting")

# 👇 Streamlit form to take user input
with st.form("single_job_form"):
    title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    location = st.text_input("Location")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    submit_button = st.form_submit_button("Check This Job")  # ✅ this creates submit_button!

# 👇 This must be *outside* the form
if submit_button:
    input_df = pd.DataFrame([{
        "title": title,
        "company_profile": company,
        "location": location,
        "description": description,
        "requirements": requirements
    }])

    try:
        # Preprocess using your vectorizer
        X_input = preprocess(input_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.success("✅ Prediction Complete!")
        st.markdown(f"🧠 **Prediction:** { 'Fake' if pred == 1 else 'Real' }")
        st.markdown(f"📊 **Fraud Probability:** {prob:.2%}")

        if prob > 0.75:
            st.warning("🚨 High Risk! This job may be a scam.")
        elif prob > 0.5:
            st.info("⚠ Moderate Risk. Review carefully before applying.")
        else:
            st.success("👍 Low Risk. Likely a genuine job post.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

     