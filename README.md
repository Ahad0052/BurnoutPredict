# 🔥 Developer Burnout Predictor

A Streamlit web app that predicts developer burnout risk (Low / Medium / High) using a trained Logistic Regression model. Built as part of the **Nafath Bootcamp 2026 — DS & AI Track**.

-----

## 📸 Preview

> Fill in a developer profile → click **Predict Burnout Level** → get an instant prediction with confidence scores and actionable tips.

-----

## 🚀 Features

- **Instant burnout prediction** — Low, Medium, or High risk levels
- **Confidence breakdown** — visual probability bars for each class
- **Smart input validation** — catches impossible values (e.g. sleep + work hours > 24)
- **Input summary** — expandable table showing all values used for prediction
- **Interpretation tips** — actionable recommendations based on the result
- **Dark-themed UI** — polished custom CSS with gradient cards and color-coded results

-----

## 🧠 Model

The app loads three pre-trained files at startup:

|File                  |Description                                          |
|----------------------|-----------------------------------------------------|
|`burnout_pipeline.pkl`|Sklearn pipeline (preprocessor + Logistic Regression)|
|`feature_names.pkl`   |Ordered list of feature names expected by the model  |
|`label_encoder.pkl`   |Label encoder to decode predicted class indices      |


> ⚠️ All three `.pkl` files must be in the **same directory** as `burnout_predictor.py`.

-----

## 📋 Input Features

|Feature         |Range   |Description                                   |
|----------------|--------|----------------------------------------------|
|Age             |18–65   |Developer’s age in years                      |
|Experience      |0–40 yrs|Years of professional dev experience          |
|Daily Work Hours|1–18 hrs|Average hours worked per day                  |
|Sleep Hours     |2–12 hrs|Average hours of sleep per night              |
|Caffeine Intake |0–15    |Caffeinated drinks per day                    |
|Bugs per Day    |0–50    |Average bugs encountered daily                |
|Commits per Day |0–50    |Average code commits per day                  |
|Meetings per Day|0–15    |Average meetings attended per day             |
|Screen Time     |1–20 hrs|Total daily screen time                       |
|Exercise        |0–5 hrs |Hours of physical exercise per day            |
|Stress Level    |0–100   |Self-reported stress (0 = none, 100 = extreme)|

-----

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/developer-burnout-predictor.git
cd developer-burnout-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model files

Place the following files in the project root:

```
burnout_pipeline.pkl
feature_names.pkl
label_encoder.pkl
```

### 4. Run the app

```bash
streamlit run burnout_predictor.py
```

The app will open at `http://localhost:8501`.

-----

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
```

Create a `requirements.txt` with the above, or install manually:

```bash
pip install streamlit pandas scikit-learn joblib
```

-----

## 📁 Project Structure

```
developer-burnout-predictor/
├── burnout_predictor.py   # Main Streamlit app
├── burnout_pipeline.pkl   # Trained model pipeline
├── feature_names.pkl      # Feature name ordering
├── label_encoder.pkl      # Class label encoder
├── requirements.txt       # Python dependencies
└── README.md
```

-----

## ⚠️ Disclaimer

This tool is built for **educational purposes** as part of a data science bootcamp project. Predictions are based on a machine learning model trained on synthetic/sample data and should **not** be used as a substitute for professional mental health advice.

-----

## 👤 Author

Built for the **Nafath Bootcamp 2026 — DS & AI Track**.