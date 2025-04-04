import numpy as np
import streamlit as st
import pickle
import base64

# Function to add missing labels to a label encoder
def add_missing_labels(encoder, new_labels):
    existing_labels = set(encoder.classes_)
    updated_labels = sorted(existing_labels.union(new_labels))
    encoder.classes_ = np.array(updated_labels)

# Load the model and label encoders
with open("crop_yield_prediction_model.pkl", "rb") as file:
    data = pickle.load(file)
    model = data['model']
    label_encoders = data['label_encoders']
    scaler = data['scaler']

# Ensure label encoders have all possible labels
soil_types = ['Loamy', 'Alluvial']
add_missing_labels(label_encoders['Soil_Type'], soil_types)
crop_recommendations = {
    'Loamy': ['Wheat', 'Rice', 'Bajra'],
    'Alluvial': ['Wheat', 'Rice']
}
all_crops = [crop for crops in crop_recommendations.values() for crop in crops]
add_missing_labels(label_encoders['Crop_Type'], all_crops)

# Conversion functions
def convert_hectares_to_acres(size_hectares):
    return size_hectares * 2.47

def calculate_sacks(yield_kg, sack_size_kg=50):
    return yield_kg / sack_size_kg

# Add custom CSS for background
def add_background(image_path):
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode('utf-8')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add background image
add_background("crops-growing-in-thailand.jpg")

# Streamlit app design
st.title("🌾 Farmer-Friendly Crop Yield Predictor")

# Inputs
soil_type = st.selectbox("Select Soil Type:", list(crop_recommendations.keys()))
st.write(f"Recommended Crops for {soil_type}: {', '.join(crop_recommendations[soil_type])}")
crop_type = st.selectbox("Select Crop Type:", crop_recommendations[soil_type])
year = st.number_input("Enter Year:", min_value=2000, max_value=2030, value=2023, step=1)
irrigation_area = st.number_input("Enter Irrigated Area (in hectares):", min_value=0.0, value=0.0, step=0.1)
irrigation_area_acres = convert_hectares_to_acres(irrigation_area)
st.write(f"Irrigated Area: {irrigation_area_acres:.2f} acres")

default_msp = 100.0
msp = st.number_input("Enter Minimum Support Price (MSP) (₹/kg):", min_value=0.0, value=default_msp, step=1.0)
sack_size = st.number_input("Enter Sack Size (kg):", min_value=50, max_value=100, value=50, step=5)

# Predict button
if st.button('Predict Yield'):
    try:
        # Encode the categorical variables
        crop_type_encoded = label_encoders['Crop_Type'].transform([crop_type])[0]
        soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])[0]

        # Scale the irrigation area
        irrigation_area_scaled = scaler.transform([[irrigation_area]])[0][0]

        # Prepare input features for the model
        input_features = np.array([[year, crop_type_encoded, soil_type_encoded, irrigation_area_scaled]])

        # Predict yield
        predicted_yield_per_ha = model.predict(input_features)[0]
        total_yield_kg = irrigation_area * predicted_yield_per_ha
        total_revenue = total_yield_kg * msp
        total_sacks = calculate_sacks(total_yield_kg, sack_size)

        # Display results
        st.markdown(
            f"""
            <p style="color:black; font-weight:bold;">
                Predicted Yield: {total_yield_kg:.2f} kg
            </p>
            <p style="color:black; font-weight:bold;">
                Equivalent Sacks: {total_sacks:.0f} sacks
            </p>
            <p style="color:black; font-weight:bold;">
                Estimated Revenue: ₹{total_revenue:,.2f}
            </p>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
