#######################
# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.dates as mdates

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

import joblib

from PIL import Image

#######################
# Page configuration
st.set_page_config(
    page_title="Seattle Weather Prediction", # Replace this with your Project's Title
    page_icon=":sun_behind_rain_cloud:", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Seattle Weather Prediction')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Butial, Aivann Paul O.\n2. Lim, Evan Vincent B.\n3. Lim, Kyle Hendrik L.\n4. Ongtangco, Randolph Joshua M.\n5. Tan, Gabriel Christian D.")

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1xwaCdEhWPi_2sUwpqr9Mp2uCdcOyj45_#scrollTo=MQo1i9FqwRfd)")
    st.markdown(":file_cabinet: [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
#######################
# Data

# Load data
df = pd.read_csv("data/seattle-weather.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("IRIS Flower Dataset")
    st.write("")

    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    
    col_pred = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            show_weather_conditions = st.checkbox('Show Weather Conditions')
            
            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:
                st.session_state.clear = True

    with col_pred[1]:
        st.markdown("#### 🌲🌲🌲 Random Forest Classifier")

        # Input boxes for the weather features
        precipitation = st.number_input('Precipitation (mm)', min_value=0.0, max_value=100.0, step=0.1, key='precipitation', value=0.0 if st.session_state.clear else st.session_state.get('precipitation', 0.0))
        temp_max = st.number_input('Max Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1, key='temp_max', value=0.0 if st.session_state.clear else st.session_state.get('temp_max', 0.0))
        temp_min = st.number_input('Min Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1, key='temp_min', value=0.0 if st.session_state.clear else st.session_state.get('temp_min', 0.0))
        wind = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=100.0, step=0.1, key='wind', value=0.0 if st.session_state.clear else st.session_state.get('wind', 0.0))
        
        weather_options = ['drizzle', 'rain', 'sun', 'snow', 'fog']
        weather = st.selectbox('Weather Condition', weather_options)

        # Button to detect weather based on input
        if st.button('Detect', key='detect_weather'):
            # Prepare the input data for prediction
            X_new = pd.DataFrame({
                "precipitation": [precipitation],
                "temp_max": [temp_max],
                "temp_min": [temp_min],
                "wind": [wind],
            })
            
            # Load the RandomForest model
            rfr_classifier = joblib.load('random_forest_classifier_weather.joblib')

            # Make a prediction using the RandomForest model
            rfr_prediction = rfr_classifier.predict(X_new)

            # Prepare the output data for the table
            prediction_data = {
                "Precipitation (mm)": [precipitation],
                "Max Temp (°C)": [temp_max],
                "Min Temp (°C)": [temp_min],
                "Wind Speed (km/h)": [wind],
                "Predicted Weather": [rfr_prediction[0]]
            }

            # Create a DataFrame to display predictions
            prediction_df = pd.DataFrame(prediction_data)

            # Display the prediction result in a table format
            st.subheader("Predicted Weather")
            st.write(prediction_df)

    # Optionally, show dataset and specific weather conditions
    if show_dataset:
        st.subheader("Dataset")
        st.dataframe(df, use_container_width=True, hide_index=True)

    if show_weather_conditions:
        st.subheader("Weather Conditions Samples")
        # Display some sample weather conditions (if available in the dataset)
        st.dataframe(df.sample(5), use_container_width=True, hide_index=True)

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here