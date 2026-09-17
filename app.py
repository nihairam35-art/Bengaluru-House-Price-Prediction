import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("House_Price_Prediction.pkl", "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("Bengaluru_House_Data.csv")

    return data


# ============================================================
# LOAD FILES
# ============================================================

try:

    model = load_model()
    data = load_data()

except Exception as e:

    st.error("Error loading the model or dataset.")

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Bengaluru House Price Predictor")

st.write(
    "Predict the approximate price of a house in Bengaluru "
    "using a Machine Learning model."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    This application predicts Bengaluru house prices
    using Machine Learning.

    **Input Features**

    • Location  
    • Total Square Feet  
    • Bedrooms  
    • Bathrooms  
    • Balcony
    """
)

st.sidebar.divider()

st.sidebar.info(
    "The predicted price is an estimate based on the trained Machine Learning model."
)


# ============================================================
# GET LOCATIONS
# ============================================================

if "location" in data.columns:

    locations = sorted(
        data["location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    locations = []


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Enter Property Details")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    location = st.selectbox(
        "📍 Location",
        locations
    )

    total_sqft = st.number_input(
        "📐 Total Square Feet",
        min_value=300.0,
        max_value=10000.0,
        value=1200.0,
        step=50.0
    )

    bhk = st.number_input(
        "🛏️ Number of Bedrooms (BHK)",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    bath = st.number_input(
        "🚿 Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    balcony = st.number_input(
        "🌿 Number of Balconies",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame({
            "location": [location],
            "total_sqft": [total_sqft],
            "bath": [bath],
            "balcony": [balcony],
            "bhk": [bhk]
        })


        # ----------------------------------------------------
        # DISPLAY INPUT
        # ----------------------------------------------------

        st.subheader("Your Property")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # PREPARE INPUT FOR MODEL
        # ----------------------------------------------------

        input_encoded = pd.get_dummies(
            input_data,
            columns=["location"]
        )


        # ----------------------------------------------------
        # MATCH TRAINING FEATURES
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            model_features = model.feature_names_in_

            input_encoded = input_encoded.reindex(
                columns=model_features,
                fill_value=0
            )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(input_encoded)

        price = prediction[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("🏠 Estimated House Price")


        result1, result2, result3 = st.columns(3)


        with result1:

            st.metric(
                "Price",
                f"₹ {price:.2f} Lakhs"
            )


        with result2:

            st.metric(
                "Price in Crores",
                f"₹ {price / 100:.2f} Cr"
            )


        with result3:

            price_per_sqft = (
                price * 100000
            ) / total_sqft

            st.metric(
                "Approx. Price / Sqft",
                f"₹ {price_per_sqft:,.0f}"
            )


        st.divider()


        # ----------------------------------------------------
        # PROPERTY SUMMARY
        # ----------------------------------------------------

        st.subheader("📋 Property Summary")


        summary1, summary2 = st.columns(2)


        with summary1:

            st.write(
                f"**📍 Location:** {location}"
            )

            st.write(
                f"**📐 Area:** {total_sqft:,.0f} sqft"
            )

            st.write(
                f"**🛏️ Bedrooms:** {bhk}"
            )


        with summary2:

            st.write(
                f"**🚿 Bathrooms:** {bath}"
            )

            st.write(
                f"**🌿 Balconies:** {balcony}"
            )


    except Exception as e:

        st.error("Prediction failed.")

        st.write(
            "The model's training features and the input features "
            "may not match."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bengaluru House Price Prediction | Machine Learning Project"
)