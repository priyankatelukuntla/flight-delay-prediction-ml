# ✈️ Flight Arrival Delay Prediction

An end-to-end Machine Learning regression project that predicts expected **flight arrival delay in minutes** using historical flight information.

## 🚀 Live Application

https://flight-delay-prediction-ml-yhvbxsttcbcfdtr4skbruj.streamlit.app/

## 🎯 Objective

Predict flight arrival delay using information available before departure, including airline, airports, states, scheduled times, flight date, distance, and distance group.

The task is formulated as a **regression problem**.

## 📂 Project Structure

```text
flight-delay-prediction-ml/
├── app.py
├── requirements.txt
├── runtime.txt
├── data/
│   └── airport_state_mapping.csv
├── models/
│   ├── linear_regression_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   └── flight_delay_xgboost_pipeline.pkl
└── README.md
```

## 📊 Dataset

Historical flight data covering **2018–2022** is used for model development.

The application also uses an airport-to-state mapping file so that airport selections correspond to the selected state.

### Main Features

- Year
- Quarter
- Month
- DayofMonth
- DayOfWeek
- Marketing_Airline_Network
- Operating_Airline
- Origin
- OriginState
- Dest
- DestState
- CRSDepTime
- CRSArrTime
- CRSElapsedTime
- Distance
- DistanceGroup
- DepHour
- ArrHour

## 🔧 Machine Learning Workflow

```text
Historical Flight Data
        ↓
Data Cleaning & Preparation
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Streamlit Deployment
```

## 🤖 Models Used

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

The preprocessing and trained models are stored as pipelines and loaded by the Streamlit application.

## 📈 Model Evaluation

| Model | MAE (minutes) | RMSE (minutes) | R² Score |
|---|---:|---:|---:|
| Linear Regression | 23.66 | 40.02 | 0.063 |
| Random Forest | 22.25 | 38.48 | 0.133 |
| **XGBoost** | **21.86** | **37.92** | **0.158** |

### 🏆 Best Model

**XGBoost** achieved the lowest MAE and RMSE among the evaluated models.

- MAE: **21.86 minutes**
- RMSE: **37.92 minutes**
- R²: **0.158**

The relatively low R² indicates that substantial variation in arrival delay remains unexplained by the current feature set.

## 🖥️ Streamlit Application

The application provides:

- Model selection
- State-dependent airport selection
- Flight date selection
- Scheduled departure and arrival time inputs
- Scheduled elapsed time
- Distance and distance group
- Arrival-delay prediction
- Delay categorization
- Model performance comparison
- XGBoost feature importance

### Delay Categories

- 🟢 On Time / Early
- 🟡 Minor Delay
- 🟠 Moderate Delay
- 🔴 Significant Delay

## 🧠 Key Learning Outcomes

- End-to-end ML project development
- Regression model comparison
- Date/time feature engineering
- Categorical encoding
- MAE, RMSE and R² evaluation
- Feature-importance analysis
- Saving/loading ML pipelines
- Streamlit application development
- Streamlit Cloud deployment

## ⚠️ Limitations

Predictions are estimates, not guarantees. Actual delays can also depend on factors such as weather, air traffic, airport operations, aircraft availability, crew issues, and unexpected disruptions.

The current R² of **0.158** also shows that the available features explain only part of the variation in arrival delay.

## 🛠️ Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Joblib · Streamlit · Git/GitHub

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

```text
GitHub Repository
       ↓
Streamlit Cloud
       ↓
Install Dependencies
       ↓
Load Saved ML Pipelines
       ↓
Run app.py
       ↓
Live Application
```

## 📌 Future Improvements

- Hyperparameter tuning
- Cross-validation
- More detailed EDA
- Weather and airport operational features
- SHAP-based prediction explanations
- Model monitoring
- Additional business insights
- Further input validation and UI improvements
