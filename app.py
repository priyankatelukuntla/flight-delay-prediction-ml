import streamlit as st
import pandas as pd
import joblib
from datetime import date, time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Arrival Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Section spacing */
    h2 {
        margin-top: 1.5rem;
    }

    h3 {
        margin-top: 1rem;
    }

    /* Prediction result */
    .prediction-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .prediction-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .prediction-label {
        font-size: 0.95rem;
        color: #64748b;
    }

    /* Small information boxes */
    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    /* Keep feature importance section compact */
    .feature-section {
        max-width: 950px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

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


# ============================================================
# LOAD AIRPORT-STATE MAPPING
# ============================================================

@st.cache_data
def load_airport_mapping():

    mapping = pd.read_csv(
        "data/airport_state_mapping.csv",
        header=0,
        delimiter=","
    )

    # If the CSV was loaded as a single column,
    # split it manually using commas.
    if len(mapping.columns) == 1 and "," in mapping.columns[0]:

        column_name = mapping.columns[0]

        mapping = mapping[column_name].str.split(
            ",",
            expand=True
        )

        mapping.columns = ["State", "Airport", "Type"]

    else:
        mapping.columns = mapping.columns.str.strip()

    # Clean values
    mapping["State"] = mapping["State"].astype(str).str.strip()
    mapping["Airport"] = mapping["Airport"].astype(str).str.strip()
    mapping["Type"] = mapping["Type"].astype(str).str.strip()

    return mapping

# LOAD THE MAPPING FIRST
airport_mapping = load_airport_mapping()

# ============================================================
# CLEAN AIRPORT MAPPING
# ============================================================

# Remove accidental spaces from column names
airport_mapping.columns = (
    airport_mapping.columns
    .str.strip()
)

# Remove accidental spaces from values
for column in ["State", "Airport", "Type"]:

    if column in airport_mapping.columns:

        airport_mapping[column] = (
            airport_mapping[column]
            .astype(str)
            .str.strip()
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

model_performance = pd.DataFrame(
    {
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
    }
)


# ============================================================
# HEADER
# ============================================================

st.title("✈️ Flight Arrival Delay Prediction")

st.markdown(
    """
    Predict the expected **arrival delay in minutes** using
    machine learning models trained on historical flight data.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    ["XGBoost", "Random Forest", "Linear Regression"],
    index=0
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

st.sidebar.markdown("---")

st.sidebar.caption(
    "Flight Arrival Delay Prediction"
)

st.sidebar.caption(
    "Machine Learning Regression Project"
)


# ============================================================
# DROPDOWN OPTIONS
# ============================================================

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


states = sorted(
    airport_mapping["State"]
    .dropna()
    .unique()
    .tolist()
)


# ============================================================
# FLIGHT INFORMATION
# ============================================================

st.subheader("🛫 Flight Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

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

    origin_state = st.selectbox(
        "Origin State",
        states,
        key="origin_state"
    )

    origin_airports = sorted(
        airport_mapping[
            airport_mapping["State"] == origin_state
        ]["Airport"].dropna().unique()
    )

    origin_airport = st.selectbox(
        "Origin Airport",
        origin_airports,
        key="origin_airport"
    )


with col2:

    destination_state = st.selectbox(
        "Destination State",
        states,
        key="destination_state"
    )

    destination_airports = sorted(
        airport_mapping[
            airport_mapping["State"] == destination_state
        ]["Airport"].dropna().unique()
    )

    destination_airport = st.selectbox(
        "Destination Airport",
        destination_airports,
        key="destination_airport"
    )

    flight_date = st.date_input(
        "Flight Date",
        value=date(2022, 1, 15)
    )


# ============================================================
# SCHEDULE INFORMATION
# ============================================================

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

   # Convert scheduled times to minutes from midnight
    departure_minutes = (
        departure_time.hour * 60
        + departure_time.minute
    )

    arrival_minutes = (
        arrival_time.hour * 60
        + arrival_time.minute
    )

    # Calculate scheduled elapsed time
    scheduled_elapsed_time = (
        arrival_minutes - departure_minutes
    )

    # Handle flights arriving after midnight
    if scheduled_elapsed_time < 0:
        scheduled_elapsed_time += 24 * 60

    elapsed_time = st.number_input(
    "Scheduled Elapsed Time (minutes)",
    value=scheduled_elapsed_time,
    disabled=True
    )


# ============================================================
# FLIGHT CHARACTERISTICS
# ============================================================

st.subheader("📍 Flight Characteristics")

col1, col2 = st.columns(2)


with col1:

    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        max_value=6000,
        value=500,
        step=10,
        help="Flight distance in miles."
    )


with col2:

    distance_group = st.number_input(
        "Distance Group",
        min_value=1,
        max_value=11,
        value=3,
        step=1,
        help="Distance group used by the training dataset."
    )


# ============================================================
# PREPARE INPUT FEATURES
# ============================================================

# ------------------------------------------------------------
# Convert Time to HHMM format
# ------------------------------------------------------------

crs_dep_time = (
    departure_time.hour * 100
    + departure_time.minute
)

crs_arr_time = (
    arrival_time.hour * 100
    + arrival_time.minute
)


# ------------------------------------------------------------
# Derived time features
# ------------------------------------------------------------

dep_hour = crs_dep_time // 100

arr_hour = crs_arr_time // 100


# ------------------------------------------------------------
# Date features
# ------------------------------------------------------------

year = flight_date.year

month = flight_date.month

day_of_month = flight_date.day

quarter = (month - 1) // 3 + 1

day_of_week = flight_date.weekday() + 1


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_clicked = st.button(
    "🔮 Predict Arrival Delay",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if origin_airport is None or destination_airport is None:
        st.error(
            "Please select valid origin and destination airports."
        )

    elif origin_airport == destination_airport:

        st.warning(
            "⚠️ Origin and destination airports are the same. "
            "Please select different airports."
        )

    else:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
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
                    origin_airport
                ],

                "OriginState": [
                    origin_state
                ],

                "Dest": [
                    destination_airport
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
            }
        )


        # ----------------------------------------------------
        # Selected model
        # ----------------------------------------------------

        model = models[selected_model]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # Prevent negative delay
        # ----------------------------------------------------

        prediction = max(0, prediction)


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # Predicted delay
        # ----------------------------------------------------

        with result_col1:

            st.metric(
                "Predicted Arrival Delay",
                f"{prediction:.1f} min"
            )


        # ----------------------------------------------------
        # Delay category
        # ----------------------------------------------------

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


        st.caption(
            f"Prediction generated using **{selected_model}**."
        )


        # ====================================================
        # MODEL INPUT
        # ====================================================

        with st.expander("🔍 View Model Input"):

            st.dataframe(
                input_data,
                width="stretch",
                hide_index=True
            )


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()

st.subheader("📊 Model Performance Comparison")

st.markdown(
    """
    The models were evaluated on the test dataset using
    **Mean Absolute Error (MAE)**,
    **Root Mean Squared Error (RMSE)**,
    and **R² Score**.
    """
)


# ------------------------------------------------------------
# Performance table
# ------------------------------------------------------------

st.dataframe(
    model_performance,
    width=850,
    hide_index=True,
    column_config={

        "Model": st.column_config.TextColumn(
            "Model",
            width="large"
        ),

        "MAE (minutes)": st.column_config.NumberColumn(
            "MAE (minutes)",
            format="%.2f"
        ),

        "RMSE (minutes)": st.column_config.NumberColumn(
            "RMSE (minutes)",
            format="%.2f"
        ),

        "R² Score": st.column_config.NumberColumn(
            "R² Score",
            format="%.3f"
        )
    }
)


# ------------------------------------------------------------
# Best model
# ------------------------------------------------------------

best_model_row = model_performance.loc[
    model_performance["MAE (minutes)"].idxmin()
]

best_model_name = best_model_row["Model"]

best_mae = best_model_row["MAE (minutes)"]

best_rmse = best_model_row["RMSE (minutes)"]

best_r2 = best_model_row["R² Score"]


st.success("🏆 Recommended Model: XGBoost")

st.markdown(
    "XGBoost achieved the best performance among the evaluated models "
    "with an MAE of 21.86 minutes."
)


# ============================================================
# MODEL PERFORMANCE VISUALIZATION
# ============================================================

st.subheader("📈 Model Performance Visualization")

chart_col1, chart_col2 = st.columns(2)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

with chart_col1:

    st.markdown("### MAE Comparison")

    mae_chart = (
        model_performance
        .set_index("Model")["MAE (minutes)"]
    )

    st.bar_chart(
        mae_chart,
        x_label="Model",
        y_label="MAE (minutes)"
    )


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

with chart_col2:

    st.markdown("### RMSE Comparison")

    rmse_chart = (
        model_performance
        .set_index("Model")["RMSE (minutes)"]
    )

    st.bar_chart(
        rmse_chart,
        x_label="Model",
        y_label="RMSE (minutes)"
    )


# ------------------------------------------------------------
# R2
# ------------------------------------------------------------

st.markdown("### R² Score Comparison")

r2_chart = (
    model_performance
    .set_index("Model")["R² Score"]
)

st.bar_chart(
    r2_chart,
    x_label="Model",
    y_label="R² Score"
)


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("🔍 XGBoost Feature Importance")

st.markdown(
    """
    Feature importance indicates which transformed features
    contributed most to the XGBoost model's predictions.
    """
)


# ------------------------------------------------------------
# Get XGBoost pipeline
# ------------------------------------------------------------

xgb_pipeline = models["XGBoost"]

xgb_preprocessor = (
    xgb_pipeline.named_steps["preprocessor"]
)

xgb_model = (
    xgb_pipeline.named_steps["model"]
)


# ------------------------------------------------------------
# Get transformed feature names
# ------------------------------------------------------------

feature_names = (
    xgb_preprocessor
    .get_feature_names_out()
)


# ------------------------------------------------------------
# Create feature importance dataframe
# ------------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": xgb_model.feature_importances_
    }
)


# ------------------------------------------------------------
# Sort and select top 20
# ------------------------------------------------------------

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(20)
)


# ============================================================
# CLEAN FEATURE NAMES
# ============================================================

def clean_feature_name(feature):

    feature = feature.replace(
        "categorical__",
        ""
    )

    feature = feature.replace(
        "numerical__",
        ""
    )

    replacements = {

        "Marketing_Airline_Network_":
            "Marketing Airline - ",

        "Operating_Airline_":
            "Operating Airline - ",

        "OriginState_":
            "Origin State - ",

        "DestState_":
            "Destination State - ",

        "Origin_":
            "Origin Airport - ",

        "Dest_":
            "Destination Airport - ",

        "CRSDepTime":
            "Scheduled Departure Time",

        "CRSArrTime":
            "Scheduled Arrival Time",

        "CRSElapsedTime":
            "Scheduled Elapsed Time",

        "DistanceGroup":
            "Distance Group",

        "Distance":
            "Distance",

        "DepHour":
            "Departure Hour",

        "ArrHour":
            "Arrival Hour",

        "DayofMonth":
            "Day of Month",

        "DayOfWeek":
            "Day of Week",

        "Year":
            "Year",

        "Quarter":
            "Quarter",

        "Month":
            "Month"
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


# ============================================================
# TOP 20 FEATURES TABLE
# ============================================================

st.markdown("### Top 20 Features")

st.caption(
    "Scroll inside the table to view all features."
)


# ------------------------------------------------------------
# Fixed width keeps table compact
# ------------------------------------------------------------

st.dataframe(
    feature_importance,
    width=900,
    height=350,
    hide_index=True,

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


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

st.markdown("### Feature Importance Distribution")

importance_chart = (
    feature_importance
    .sort_values("Importance")
    .set_index("Feature")["Importance"]
)


st.bar_chart(
    importance_chart,
    x_label="Feature",
    y_label="Importance",
    height=450
)


# ============================================================
# HOW THE PREDICTION WORKS
# ============================================================

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


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.divider()

st.subheader("📊 Project Overview")

overview_col1, overview_col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT
# ------------------------------------------------------------

with overview_col1:

    st.markdown("### 🎯 Objective")

    st.write(
        """
        The objective of this project is to predict the expected
        **flight arrival delay in minutes** using historical
        flight information and machine learning regression models.
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


# ------------------------------------------------------------
# RIGHT
# ------------------------------------------------------------

with overview_col2:

    st.markdown("### 📌 Prediction Type")

    st.write(
        """
        **Regression**

        The model predicts a continuous numerical value
        representing the expected arrival delay in minutes.
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


# ============================================================
# BEST PERFORMING MODEL
# ============================================================

st.markdown("### 🏆 Best Performing Model")

st.success(
    f"""
    **{best_model_name}** achieved the best performance among
    the evaluated models.

    **MAE:** {best_mae:.2f} minutes  
    **RMSE:** {best_rmse:.2f} minutes  
    **R² Score:** {best_r2:.3f}
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ✈️ Flight Arrival Delay Prediction |
        Machine Learning Regression Project
    </div>
    """,
    unsafe_allow_html=True
)