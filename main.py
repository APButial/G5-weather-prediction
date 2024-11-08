#######################
# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pickle

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
    page_icon="🌦️", # You may replace this with a custom icon or emoji related to your project
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
    st.markdown("A Streamlit dashboard highlighting the results of of training Random Forest Classification model using the Weather Prediction dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/ananthr1/weather-prediction)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1xwaCdEhWPi_2sUwpqr9Mp2uCdcOyj45_#scrollTo=MQo1i9FqwRfd)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/APButial/G5-weather-prediction.git)")
#######################
# Data

# Load data
df = pd.read_csv("./data/seattle-weather.csv")

#######################

# Import models


#######################

# Plots

# `key` parameter is used to update the plot when the page is refreshed

def feature_importance_plot(feature_importance_df, width, height, key):
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h',
        color='Feature',
        color_discrete_sequence=['cyan','indianred','yellow','lightgreen']
    )

    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")

def confusion_matrix():
    with open('./resource/confusion_matrix.pkl', 'rb') as f:
        cm = pickle.load(f)

    plt.figure(figsize=(1, 1))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['drizzle','fog','rain','snow','sun']).plot(cmap=plt.cm.Blues)
    plt.title('Weather Prediction Confusion Matrix')
    plt.xlabel('Predicted Weather')
    plt.ylabel('Actual Weather')

    st.pyplot(plt, use_container_width=False)
########################

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
# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # Initial data overview section with explicit explanations
    st.markdown("""
    ### Initial Data Overview
    The dataset includes **weather observations** in Seattle over a period from 2012 to 2015. It consists of columns representing:
    - `date`: Date of observation
    - `precipitation`: Precipitation in inches
    - `temp_max` and `temp_min`: Maximum and minimum temperatures in Celsius
    - `wind`: Average wind speed in mph
    - `weather`: Categorical weather condition (e.g., drizzle, fog, rain, snow, sun)

    #### Observations:
    - **No missing values** were found in the dataset.
    - The dataset includes **balanced occurrences** of most weather categories, though **snow** has fewer entries.
    - **No duplicate rows** are present.
    """)

    # Checking for missing values and column types manually
    st.subheader("Null Values and Data Types")
    # Display data types and null counts
    st.write(pd.DataFrame({'Data Type': df.dtypes, 'Null Values': df.isna().sum()}))

    # Distribution of weather conditions
    st.subheader("Weather Condition Distribution")
    colors = ['skyblue', 'yellow', 'lightgreen', 'salmon', 'orange']
    fig, ax = plt.subplots()
    ax.pie(df['weather'].value_counts(), labels=df['weather'].value_counts().index, autopct='%1.1f%%', colors=colors)
    plt.title('Weather Occurrences in Seattle (2012-2015)')
    st.pyplot(fig)

    st.markdown("""
    The pie chart illustrates the distribution of different weather conditions. **Rain** and **sun** are the most common conditions, while **snow** is relatively rare. This imbalance will be addressed later in the pipeline with oversampling techniques.
    """)

    # Convert 'date' column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        st.markdown("""
        ### Date Conversion
        - The `date` column has been converted to **datetime format** for better time-based analysis.
        """)

    # Calculate average temperature
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    st.markdown("""
    ### Calculating Average Temperature
    - A new column `temp_avg` was created to represent the **average daily temperature** based on `temp_max` and `temp_min`.
    """)
    st.dataframe(df[['temp_max', 'temp_min', 'temp_avg']].head(), use_container_width=True, hide_index=True)

    # Encoding categorical column
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    st.markdown("""
    ### Encoding Weather Condition
    - The `weather` column, originally categorical, has been encoded into numerical values using `LabelEncoder`.
    - The encoded values will be used as labels for training the machine learning model.
    """)
    st.dataframe(df[['weather', 'weather_encoded']].drop_duplicates(), use_container_width=True, hide_index=True)

    # Train-Test Split
    st.subheader("Train-Test Split")
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
    y = df['weather_encoded']

    st.code("""
    # Select features and target variable
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
    y = df['weather_encoded']
    """)

    st.markdown("""
    The selected features for model training are:
    - **precipitation**: Amount of rainfall (in inches)
    - **temp_max** and **temp_min**: Maximum and minimum temperatures (in Celsius)
    - **wind**: Wind speed (in mph)
    
    The label is the **encoded weather condition** (`weather_encoded`).
    """)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.code("""
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    """)

    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)

    st.markdown("""
    After splitting the dataset into `training` and `test` sets, we proceed with **oversampling** to address the class imbalance.
    """)

    # Applying Borderline SMOTE for Oversampling
    sampler = BorderlineSMOTE(random_state=42, sampling_strategy='auto', kind='borderline-2')
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    st.markdown("""
    ### Applying Borderline SMOTE for Class Imbalance
    - **Borderline SMOTE** oversampling technique was applied to the training data to balance the class distribution, especially for rarer conditions like `snow`.
    - This step ensures that the model does not bias towards the more frequent classes.
    """)

    # Visualization of Resampled Data
    resampled_dist = pd.DataFrame.from_dict(Counter(y_train), orient='index', columns=['Count'])
    resampled_dist.index.name = 'Class'
    resampled_dist.index = le_weather.inverse_transform(resampled_dist.index)

    st.subheader("Resampled Weather Distribution")
    st.dataframe(resampled_dist, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(Counter(y_train).values(), labels=[le_weather.classes_[i] for i in Counter(y_train).keys()], autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Weather Distribution After Borderline SMOTE')
    st.pyplot(fig)

    st.markdown("""
    After oversampling, each weather condition is more evenly represented in the training data, which will help the model learn to classify each condition more accurately.
    """)



# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here
    st.subheader("🌲🌲🌲 Random Forest Classifier")
    st.markdown("""
                A classifier model that uses multiple decision tree classifiers on dataset sub-samples. 
                Results of the classifiers are then averaged to determine the predictive accuracy of the model.\n
                `Reference:` https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                """)
    st.subheader("🏋️‍♂️ Training the Model")
    st.code("""
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            """)
    st.subheader("✅ Model Evaluation")
    st.code("""
            # Prediction
            y_pred = model.predict(X_test)

            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le_weather.classes_)

            print(f"Accuracy: {accuracy}")
            print(f"Classification Report: {report}")
            """)
    st.code("""
            Accuracy: 79.73%\n
            Classification Report:\n
                          precision    recall  f1-score   support
                 drizzle       0.14      0.14      0.14        14
                     fog       0.24      0.25      0.25        32
                    rain       0.97      0.91      0.94       192
                    snow       0.43      0.38      0.40         8
                     sun       0.80      0.84      0.82       193

                accuracy                           0.80       439
               macro avg       0.52      0.50      0.51       439
            weighted avg       0.80      0.80      0.80       439
            """)
    st.subheader("🌡️ Feature Importance")
    st.code("""
            feature_importance = pd.Series((model.feature_importances_)*100, index=X_train.columns)
            feature_importance
            """)
    feature_importance = {
        'Feature': ['precipitation','temp_max','temp_min','wind'],
        'Importance': [27.307511, 29.115227, 25.766615, 17.810646]
    }

    feature_importance_df = pd.DataFrame(feature_importance)
    st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)
    feature_importance_plot(feature_importance_df, 500, 500, 2)
    st.write("""The feature importance of the Random Forest Classifier model indicates 
             `temp_max` had the most influence on the model's weather prediction.""")
    
    st.subheader("❓ Confusion Matrix")
    confusion_matrix()

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    
    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here