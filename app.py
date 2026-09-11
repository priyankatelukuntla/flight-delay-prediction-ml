import streamlit as st
import pandas as pd
import joblib
from datetime import date, time


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    models = {}

    models["Linear Regression"] = joblib.load(
        "models/linear_regression_pipeline.pkl"
    )

    models["Random Forest"] = joblib.load(
        "models/random_forest_pipeline.pkl"
    )

    models["XGBoost"] = joblib.load(
        "models/flight_delay_xgboost_pipeline.pkl"
    )

    return models


models = load_models()

# ==================================================
# MODEL PERFORMANCE
# ==================================================

model_performance = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE (minutes)": [
        23.66,
        22.25,
        21.86
    ],
    "RMSE (minutes)": [
        40.02,
        38.48,
        37.92
    ],
    "R² Score": [
        0.063,
        0.133,
        0.158
    ]
})

# ==================================================
# HEADER
# ==================================================

st.title("✈️ Flight Arrival Delay Prediction")

st.markdown(
    """
    Predict the expected **arrival delay in minutes** using
    Machine Learning models trained on historical flight data.
    """
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("🤖 Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Models available**

    • Linear Regression  
    • Random Forest  
    • XGBoost
    """
)


# ==================================================
# FLIGHT INFORMATION
# ==================================================

st.subheader("🛫 Flight Information")

col1, col2 = st.columns(2)


# --------------------------------------------------
# Dropdown Options
# --------------------------------------------------

airlines = [
    "AA",
    "AS",
    "B6",
    "DL",
    "F9",
    "HA",
    "NK",
    "UA",
    "WN"
]

airports = [
    "ATL",
    "DEN",
    "DFW",
    "ORD",
    "LAX",
    "JFK",
    "LAS",
    "SEA",
    "SFO",
    "PHX",
    "MCO",
    "CLT"
]

states = [
    "AL",
    "AZ",
    "CA",
    "CO",
    "FL",
    "GA",
    "IL",
    "NC",
    "NV",
    "NY",
    "TX",
    "WA"
]


# --------------------------------------------------
# Left Column
# --------------------------------------------------

with col1:

    marketing_airline = st.selectbox(
        "Marketing Airline",
        airlines,
        index=3
    )

    operating_airline = st.selectbox(
        "Operating Airline",
        airlines,
        index=3
    )

    origin = st.selectbox(
        "Origin Airport",
        airports,
        index=0
    )

    origin_state = st.selectbox(
        "Origin State",
        states,
        index=6
    )


# --------------------------------------------------
# Right Column
# --------------------------------------------------

with col2:

    destination = st.selectbox(
        "Destination Airport",
        airports,
        index=3
    )

    destination_state = st.selectbox(
        "Destination State",
        states,
        index=7
    )

    flight_date = st.date_input(
        "Flight Date",
        value=date(2022, 1, 15)
    )


# ==================================================
# SCHEDULE INFORMATION
# ==================================================

st.subheader("🕐 Schedule Information")

col1, col2, col3 = st.columns(3)


with col1:

    departure_time = st.time_input(
        "Scheduled Departure Time",
        value=time(8, 30)
    )


with col2:

    arrival_time = st.time_input(
        "Scheduled Arrival Time",
        value=time(10, 45)
    )


with col3:

    elapsed_time = st.number_input(
        "Scheduled Elapsed Time (minutes)",
        min_value=1,
        value=135
    )


# ==================================================
# FLIGHT CHARACTERISTICS
# ==================================================

st.subheader("📍 Flight Characteristics")

col1, col2 = st.columns(2)


with col1:

    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        value=606
    )


with col2:

    distance_group = st.number_input(
        "Distance Group",
        min_value=1,
        max_value=11,
        value=3
    )


# ==================================================
# PREPARE INPUT FEATURES
# ==================================================

# --------------------------------------------------
# Convert Time to HHMM Format
# --------------------------------------------------

crs_dep_time = (
    departure_time.hour * 100
    + departure_time.minute
)

crs_arr_time = (
    arrival_time.hour * 100
    + arrival_time.minute
)


# --------------------------------------------------
# Derived Time Features
# --------------------------------------------------

dep_hour = crs_dep_time // 100
arr_hour = crs_arr_time // 100


# --------------------------------------------------
# Date Features
# --------------------------------------------------

year = flight_date.year

month = flight_date.month

day_of_month = flight_date.day

quarter = (month - 1) // 3 + 1

# Dataset convention:
# Sunday = 1
# Monday = 2
# ...
# Saturday = 7

day_of_week = flight_date.weekday() + 1


# ==================================================
# PREDICTION
# ==================================================

st.divider()

if st.button(
    "🔮 Predict Arrival Delay",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------

    input_data = pd.DataFrame({

        "Year": [year],

        "Quarter": [quarter],

        "Month": [month],

        "DayofMonth": [day_of_month],

        "DayOfWeek": [day_of_week],

        "Marketing_Airline_Network": [
            marketing_airline
        ],

        "Operating_Airline": [
            operating_airline
        ],

        "Origin": [
            origin
        ],

        "OriginState": [
            origin_state
        ],

        "Dest": [
            destination
        ],

        "DestState": [
            destination_state
        ],

        "CRSDepTime": [
            crs_dep_time
        ],

        "CRSArrTime": [
            crs_arr_time
        ],

        "CRSElapsedTime": [
            elapsed_time
        ],

        "Distance": [
            distance
        ],

        "DistanceGroup": [
            distance_group
        ],

        "DepHour": [
            dep_hour
        ],

        "ArrHour": [
            arr_hour
        ]
    })


    # --------------------------------------------------
    # Get Selected Model
    # --------------------------------------------------

    model = models[selected_model]


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------
    # Prevent Negative Delay
    # --------------------------------------------------

    prediction = max(0, prediction)


    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    # --------------------------------------------------
    # Predicted Delay
    # --------------------------------------------------

    with result_col1:

        st.metric(
            "Predicted Arrival Delay",
            f"{prediction:.1f} min"
        )


    # --------------------------------------------------
    # Delay Category
    # --------------------------------------------------

    with result_col2:

        if prediction <= 0:

            status = "🟢 On Time / Early"

        elif prediction <= 15:

            status = "🟡 Minor Delay"

        elif prediction <= 60:

            status = "🟠 Moderate Delay"

        else:

            status = "🔴 Significant Delay"


        st.metric(
            "Delay Category",
            status
        )


    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    st.caption(
        f"Prediction generated using **{selected_model}**."
    )


    # ==================================================
    # VIEW MODEL INPUT
    # ==================================================

    with st.expander("🔍 View Model Input"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

# ==================================================
# MODEL PERFORMANCE COMPARISON
# ==================================================

st.divider()

st.subheader("📊 Model Performance Comparison")

st.markdown(
    """
    The models were evaluated on the test dataset using
    **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**,
    and **R² Score**.
    """
)

# Display metrics table

st.dataframe(
    model_performance,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_model = model_performance.loc[
    model_performance["MAE (minutes)"].idxmin(),
    "Model"
]

best_mae = model_performance["MAE (minutes)"].min()

st.success(
    f"🏆 **Best performing model: {best_model}** "
    f"with an MAE of **{best_mae:.2f} minutes**."
)

# ==================================================
# PERFORMANCE VISUALIZATION
# ==================================================

st.subheader("📈 Model Performance Visualization")

chart_col1, chart_col2 = st.columns(2)


# --------------------------------------------------
# MAE Comparison
# --------------------------------------------------

with chart_col1:

    st.markdown("### MAE Comparison")

    mae_chart = model_performance.set_index("Model")[
        "MAE (minutes)"
    ]

    st.bar_chart(
        mae_chart,
        x_label="Model",
        y_label="MAE (minutes)"
    )


# --------------------------------------------------
# RMSE Comparison
# --------------------------------------------------

with chart_col2:

    st.markdown("### RMSE Comparison")

    rmse_chart = model_performance.set_index("Model")[
        "RMSE (minutes)"
    ]

    st.bar_chart(
        rmse_chart,
        x_label="Model",
        y_label="RMSE (minutes)"
    )


# --------------------------------------------------
# R² Comparison
# --------------------------------------------------

st.markdown("### R² Score Comparison")

r2_chart = model_performance.set_index("Model")[
    "R² Score"
]

st.bar_chart(
    r2_chart,
    x_label="Model",
    y_label="R² Score"
)

# ==================================================
# XGBOOST FEATURE IMPORTANCE
# ==================================================

st.divider()

st.subheader("🔍 XGBoost Feature Importance")

st.markdown(
    """
    Feature importance indicates which transformed features had the
    greatest influence on the XGBoost model.
    """
)


# --------------------------------------------------
# Get XGBoost Pipeline
# --------------------------------------------------

xgb_pipeline = models["XGBoost"]

xgb_preprocessor = xgb_pipeline.named_steps["preprocessor"]

xgb_model = xgb_pipeline.named_steps["model"]


# --------------------------------------------------
# Get Transformed Feature Names
# --------------------------------------------------

feature_names = xgb_preprocessor.get_feature_names_out()


# --------------------------------------------------
# Create Feature Importance DataFrame
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.feature_importances_
})


# --------------------------------------------------
# Sort and Select Top 20
# --------------------------------------------------

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(20)
)


# --------------------------------------------------
# Clean Feature Names
# --------------------------------------------------

def clean_feature_name(feature):

    # Remove pipeline prefixes
    feature = feature.replace(
        "categorical__",
        ""
    )

    feature = feature.replace(
        "numerical__",
        ""
    )

    # Make names more readable
    replacements = {
        "Marketing_Airline_Network_": "Marketing Airline - ",
        "Operating_Airline_": "Operating Airline - ",
        "OriginState_": "Origin State - ",
        "DestState_": "Destination State - ",
        "Origin_": "Origin Airport - ",
        "Dest_": "Destination Airport - ",
        "CRSDepTime": "Scheduled Departure Time",
        "CRSArrTime": "Scheduled Arrival Time",
        "CRSElapsedTime": "Scheduled Elapsed Time",
        "DistanceGroup": "Distance Group",
        "Distance": "Distance",
        "DepHour": "Departure Hour",
        "ArrHour": "Arrival Hour",
        "DayofMonth": "Day of Month",
        "DayOfWeek": "Day of Week",
        "Year": "Year",
        "Quarter": "Quarter",
        "Month": "Month"
    }

    for old, new in replacements.items():

        if feature.startswith(old):

            feature = feature.replace(
                old,
                new,
                1
            )

            break

    return feature


feature_importance["Feature"] = (
    feature_importance["Feature"]
    .apply(clean_feature_name)
)


# --------------------------------------------------
# Display Feature Importance
# --------------------------------------------------

st.markdown("### Top 20 Features")

table_col, empty_col = st.columns([4, 1])

with table_col:

    st.dataframe(
        feature_importance,
        height=350,
        hide_index=True,
        use_container_width=True,

        column_config={

            "Feature": st.column_config.TextColumn(
                "Feature",
                width="large"
            ),

            "Importance": st.column_config.NumberColumn(
                "Importance",
                format="%.6f",
                width="small"
            )
        }
    )


# --------------------------------------------------
# Feature Importance Chart
# --------------------------------------------------

st.markdown("### Feature Importance Distribution")

importance_chart = (
    feature_importance
    .sort_values("Importance")
    .set_index("Feature")["Importance"]
)

st.bar_chart(
    importance_chart,
    x_label="Feature",
    y_label="Importance"
)

# ==================================================
# HOW THE PREDICTION WORKS
# ==================================================

st.divider()

st.subheader("⚙️ How the Prediction Works")

st.markdown(
    """
    This application uses a machine learning pipeline to predict
    **flight arrival delay in minutes** from flight, schedule,
    date, route, and distance-related information.
    """
)

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 1️⃣")
    st.markdown("**Flight Details**")
    st.caption(
        "Airline, origin, destination and state information"
    )

with step2:
    st.markdown("### 2️⃣")
    st.markdown("**Feature Engineering**")
    st.caption(
        "Date and time features are derived from the inputs"
    )

with step3:
    st.markdown("### 3️⃣")
    st.markdown("**ML Pipeline**")
    st.caption(
        "Categorical features are encoded before prediction"
    )

with step4:
    st.markdown("### 4️⃣")
    st.markdown("**Prediction**")
    st.caption(
        "The selected model predicts arrival delay in minutes"
    )

# ==================================================
# PROJECT OVERVIEW
# ==================================================

st.divider()

st.subheader("📊 Project Overview")

overview_col1, overview_col2 = st.columns(2)

with overview_col1:

    st.markdown("### 🎯 Objective")

    st.write(
        """
        The objective of this project is to predict the expected
        **flight arrival delay in minutes** using historical flight
        information and machine learning regression models.
        """
    )

    st.markdown("### 🤖 Models Used")

    st.write(
        """
        • Linear Regression  
        • Random Forest Regressor  
        • XGBoost Regressor
        """
    )


with overview_col2:

    st.markdown("### 📌 Prediction Type")

    st.write(
        """
        **Regression**

        The model predicts a continuous numerical value representing
        the expected arrival delay in minutes.
        """
    )

    st.markdown("### 📈 Evaluation Metrics")

    st.write(
        """
        • MAE — Mean Absolute Error  
        • RMSE — Root Mean Squared Error  
        • R² Score — Coefficient of Determination
        """
    )

# ==================================================
# BEST MODEL
# ==================================================

st.markdown("### 🏆 Best Performing Model")

best_model_row = model_performance.loc[
    model_performance["MAE (minutes)"].idxmin()
]

best_model_name = best_model_row["Model"]
best_mae = best_model_row["MAE (minutes)"]
best_rmse = best_model_row["RMSE (minutes)"]
best_r2 = best_model_row["R² Score"]

st.success(
    f"""
    **{best_model_name}** achieved the best performance among the
    evaluated models.

    **MAE:** {best_mae:.2f} minutes  
    **RMSE:** {best_rmse:.2f} minutes  
    **R² Score:** {best_r2:.3f}
    """
)