import streamlit as st
import plotly.express as px
import pandas as pd
import joblib

from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EstateIQ | House Price AI",
    page_icon="🏡",
    layout="wide"
)


# =====================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSOR
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Model files are stored in the repository root
MODEL_DIR = BASE_DIR


@st.cache_resource
def load_artifacts():

    model_path = MODEL_DIR / "xgboost_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    template_path = MODEL_DIR / "input_template.pkl"

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    input_template = joblib.load(template_path)

    return model, preprocessor, input_template


try:

    model, preprocessor, input_template = load_artifacts()

    model_loaded = True
    model_error = None

except Exception as e:

    model_loaded = False
    model_error = str(e)


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

[data-testid="stMetric"] {
    background-color: #17243a;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #263750;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

h1, h2, h3 {
    color: #f8fafc;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. SIDEBAR
# =====================================================

st.sidebar.title("🏡 EstateIQ")

st.sidebar.caption(
    "AI Real Estate Intelligence"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Price Prediction",
        "Model Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Model: XGBoost Regressor")

if model_loaded:

    st.sidebar.success("Model Loaded Successfully")

else:

    st.sidebar.error("Model Loading Failed")


# =====================================================
# 5. HEADER
# =====================================================

st.title("🏡 EstateIQ")

st.caption(
    "Smart Property Valuation | Powered by Machine Learning"
)

st.markdown("---")


# =====================================================
# 6. OVERVIEW DASHBOARD
# =====================================================

if page == "Overview":

    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ML Algorithm",
        "XGBoost"
    )

    col2.metric(
        "Kaggle RMSLE",
        "0.12952"
    )

    col3.metric(
        "Training Records",
        "1,460"
    )

    st.markdown("### About EstateIQ")

    st.write("""
    EstateIQ is a Machine Learning application designed
    to estimate residential property prices using
    property characteristics.

    The application uses an XGBoost Regressor trained
    on the Kaggle House Prices dataset.

    It combines feature engineering, numerical scaling,
    categorical encoding and machine learning to
    generate property price estimates.
    """)

    st.markdown("### Technology Stack")

    col1, col2, col3, col4 = st.columns(4)

    col1.info("Python")
    col2.info("Pandas")
    col3.info("Scikit-learn")
    col4.info("XGBoost")

    if model_loaded:

        st.success(
            "Your trained XGBoost model is ready."
        )

    else:

        st.error(
            f"Model error: {model_error}"
        )


# =====================================================
# 7. PRICE PREDICTION
# =====================================================

elif page == "Price Prediction":

    st.subheader("🏠 Property Valuation")

    st.write(
        "Enter property details to estimate its market value."
    )

    if not model_loaded:

        st.error(
            f"Model could not be loaded: {model_error}"
        )

    else:

        with st.form("house_form"):

            st.markdown("### Property Information")

            left, right = st.columns(2)

            with left:

                quality = st.slider(
                    "Overall Quality",
                    min_value=1,
                    max_value=10,
                    value=7
                )

                living_area = st.number_input(
                    "Living Area (sq ft)",
                    min_value=100,
                    max_value=10000,
                    value=1800
                )

                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=0,
                    max_value=10,
                    value=3
                )

                bathrooms = st.number_input(
                    "Full Bathrooms",
                    min_value=0,
                    max_value=10,
                    value=2
                )

            with right:

                basement = st.number_input(
                    "Basement Area (sq ft)",
                    min_value=0,
                    max_value=5000,
                    value=900
                )

                garage = st.number_input(
                    "Garage Capacity",
                    min_value=0,
                    max_value=5,
                    value=2
                )

                year = st.number_input(
                    "Year Built",
                    min_value=1872,
                    max_value=2010,
                    value=2005
                )

                neighborhoods = [
                    "Blmngtn",
                    "Blueste",
                    "BrDale",
                    "BrkSide",
                    "ClearCr",
                    "CollgCr",
                    "Crawfor",
                    "Edwards",
                    "Gilbert",
                    "IDOTRR",
                    "MeadowV",
                    "Mitchel",
                    "NAmes",
                    "NoRidge",
                    "NPkVill",
                    "NridgHt",
                    "NWAmes",
                    "OldTown",
                    "SWISU",
                    "Sawyer",
                    "SawyerW",
                    "Somerst",
                    "StoneBr",
                    "Timber",
                    "Veenker"
                ]

                neighborhood = st.selectbox(
                    "Neighborhood",
                    neighborhoods,
                    index=5
                )

            st.markdown("---")

            submit = st.form_submit_button(
                "🏡 Estimate Property Value"
            )

        # =============================================
        # PREDICTION LOGIC
        # =============================================

        if submit:

            try:

                # Copy saved input template
                input_data = input_template.copy()

                # Update user-provided features
                input_data.loc[:, "OverallQual"] = quality

                input_data.loc[:, "GrLivArea"] = living_area

                input_data.loc[:, "BedroomAbvGr"] = bedrooms

                input_data.loc[:, "FullBath"] = bathrooms

                input_data.loc[:, "TotalBsmtSF"] = basement

                input_data.loc[:, "GarageCars"] = garage

                input_data.loc[:, "YearBuilt"] = year

                input_data.loc[:, "Neighborhood"] = neighborhood

                # Update floor area features
                input_data.loc[:, "1stFlrSF"] = living_area

                input_data.loc[:, "2ndFlrSF"] = 0

                # Update engineered features
                input_data.loc[:, "TotalSF"] = (
                    input_data["TotalBsmtSF"]
                    + input_data["1stFlrSF"]
                    + input_data["2ndFlrSF"]
                )

                input_data.loc[:, "TotalBathrooms"] = (
                    input_data["FullBath"]
                    + 0.5 * input_data["HalfBath"]
                    + input_data["BsmtFullBath"]
                    + 0.5 * input_data["BsmtHalfBath"]
                )

                input_data.loc[:, "HouseAge"] = (
                    input_data["YrSold"]
                    - input_data["YearBuilt"]
                )

                input_data.loc[:, "RemodelAge"] = (
                    input_data["YrSold"]
                    - input_data["YearRemodAdd"]
                )

                # Validate input columns
                if list(input_data.columns) != list(input_template.columns):

                    raise ValueError(
                        "Input features do not match the saved template."
                    )

                # Apply saved preprocessing
                processed_data = preprocessor.transform(
                    input_data
                )

                # Generate prediction
                prediction = model.predict(
                    processed_data
                )[0]

                # Display result
                st.markdown("---")

                st.subheader(
                    "🏡 Estimated Property Value"
                )

                st.metric(
                    label="Predicted House Price",
                    value=f"${max(0, prediction):,.2f}"
                )

                st.success(
                    "Prediction generated successfully!"
                )

                st.caption(
                    "Currency: USD. This is a model estimate, "
                    "not a professional property appraisal."
                )

                st.info(
                    "Some property characteristics not included "
                    "in the form retain their default values from "
                    "the saved input template."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# =====================================================
# 8. MODEL ANALYTICS
# =====================================================

elif page == "Model Analytics":

    st.subheader("Model Performance Analytics")

    model_data = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ],

        "R2 Score": [
            0.444,
            0.8874,
            0.8933,
            0.9228
        ]

    })

    fig = px.bar(
        model_data,
        x="Model",
        y="R2 Score",
        text="R2 Score",
        title="Algorithm Performance Comparison",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#17243a",
        font_color="white",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Best Performing Model")

    st.success(
        "XGBoost — Validation R²: 0.9228"
    )

    st.caption(
        "These are local validation results. "
        "Kaggle RMSLE is a separate evaluation metric."
    )
