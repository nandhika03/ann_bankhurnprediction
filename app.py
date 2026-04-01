import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import streamlit as st

# Load model
model = load_model("C://Users//Nandhika//Documents//ANN_BankChurnPrediction//modelone.h5")

# Load pickle files
with open("C://Users//Nandhika//Documents//ANN_BankChurnPrediction//pickle_files//label_encoder_gender.pkl",'rb') as file:
    label_gender = pickle.load(file)

with open("C://Users//Nandhika//Documents//ANN_BankChurnPrediction//pickle_files//ohe_geography.pkl",'rb') as file:
    onehot_geo = pickle.load(file)

with open("C://Users//Nandhika//Documents//ANN_BankChurnPrediction//pickle_files//scaler.pkl",'rb') as file:
    scalar_stand = pickle.load(file)

# ---- Streamlit Custom Styling ----

# Load background
def get_img_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

image_64 = get_img_base64("C://Users//Nandhika//Documents//ANN_BankChurnPrediction//data//churn_bg.png")

    # Page setup
    #st.set_page_config(layout="wide")

    # Inject full custom style
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    div[data-baseweb="base-input"] > textarea {{
        min-height: 70px !important;
        max-height: 100px !important;
        background-color: #ffe6f0 !important;
        color: black !important;
        font-size: 16px !important;
        border: 1px solid #ff99cc !important;
        border-radius: 10px !important;
        padding: 8px !important;
        box-shadow: 0 0 8px rgba(255, 105, 180, 0.3);
    }}

    div.stButton > button:first-child {{
        background-color: #DE3163 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
    }}

    div.stButton > button:first-child:hover {{
        background-color: #c2214f !important;
        transform: scale(1.02);
    }}

    .result-box {{
        background-color: #fff0f5;
        color: black;
        padding: 12px;
        border-radius: 10px;
        font-size: 15px;
        box-shadow: 0 0 6px rgba(255,182,193,0.3);
        margin-bottom: 20px;
        white-space: pre-wrap;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* General text color (labels, titles, etc.) */
    .stText, .stMarkdown, .stLabel, .stHeader {
        color: #003366 !important;  /* dark blue text */
        font-weight: bold;
    }

    /* Input boxes, number inputs, selectboxes */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>select {
        background-color: #cce7ff !important; /* light blue box */
        border: 2px solid #0073e6 !important; /* dark blue border */
        border-radius: 8px;
        color: #003366 !important; /* dark blue text inside input */
        padding: 4px;
    }

    /* Slider styling */
    .stSlider>div>div>input[type="range"]::-webkit-slider-thumb {
        background: #0073e6 !important;  /* thumb color */
    }

    .stSlider>div>div>input[type="range"]::-moz-range-thumb {
        background: #0073e6 !important;  /* firefox thumb color */
    }

    .stSlider>div>div>input[type="range"]::-ms-thumb {
        background: #0073e6 !important;  /* IE thumb color */
    }

    .stSlider>div>div>input[type="range"] {
        accent-color: #0073e6 !important;  /* track color */
    }

    /* Buttons */
    .stButton>button {
        background-color: #99d6ff !important; /* light blue button */
        color: #003366 !important;
        border: 2px solid #0073e6 !important;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---- Streamlit App ----
st.markdown(
        "<h1 style='color: blue; text-align: center;'>Predict the possibility of churn </h1>",
        unsafe_allow_html=True
    )

# User input
geography = st.selectbox('Geography', onehot_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_data_scaled = scalar_stand.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')