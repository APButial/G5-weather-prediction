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
model = joblib.load('./model/random_forest_class.joblib')

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

def weather_occurrences(width, height, key):
    colors = ['skyblue', 'yellow', 'lightgreen', 'salmon', 'orange']

    weather_counts = df['weather'].value_counts()

    weather_fig = px.pie(
        weather_counts,
        names=weather_counts.index,
        values=weather_counts.values,
        color_discrete_sequence=colors,
    )

    # percentage val
    weather_fig.update_traces(textposition='inside', textinfo='percent+label')

    # Adjust the height and width
    weather_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(weather_fig, use_container_width=True, key=f"weather_occurrences_{key}")

def plot_average_temperature(width, height, key):
    if 'date' in df.columns:
        # If 'date' is a column, convert it to datetime and set as index
        new_df = df.copy()
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df.set_index('date', inplace=True)
    else:
        # If 'date' is not a column, print a message
        st.write("Column 'date' not found. It might already be the index or have a different name.")
        return

    new_df['temp_avg'] = (new_df['temp_max'] + new_df['temp_min']) / 2

    fig = px.line(new_df, x=new_df.index, y='temp_avg')
    fig.update_traces(line_color='red', line_width=2)

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Temperature (°C)',
        xaxis_range=['2012-01-01', '2015-12-31'],
        xaxis_tickformat='%Y',
        width=width,
        height=height
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, key=f"average_temperature_{key}")    

def plot_precipitation(width, height, key):
    if 'date' in df.columns:
        # If 'date' is a column, convert it to datetime and set as index
        new_df = df.copy()
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df.set_index('date', inplace=True)
    else:
        # If 'date' is not a column, print a message
        st.write("Column 'date' not found. It might already be the index or have a different name.")
        return

    fig = px.line(new_df, x=new_df.index, y='precipitation')

    fig.update_traces(line_color='blue', line_width=2)

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Precipitation (in)',
        xaxis_range=['2012-01-01', '2016-01-01'],
        xaxis_tickformat='%Y',
        width=width,
        height=height,
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, key=f"precipitation_over_time_{key}")

def plot_wind(width, height, key):
    if 'date' in df.columns:
        # If 'date' is a column, convert it to datetime and set as index
        new_df = df.copy()
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df.set_index('date', inplace=True)
    else:
        # If 'date' is not a column, print a message
        st.write("Column 'date' not found. It might already be the index or have a different name.")
        return

    fig = px.line(new_df, x=new_df.index, y='wind')

    fig.update_traces(line_color='green', line_width=2)

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Wind Speed (mph)',
        xaxis_range=['2012-01-01', '2016-01-01'],
        xaxis_tickformat='%Y',
        width=width,
        height=height
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, key=f"wind_speed_over_time_{key}")
    
def max_temp_scatter(width, height, key):
    fig = px.scatter(df, x='weather', y='temp_max', color='weather')
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Temperature (Celsius)',
        width=width,
        height=height
    )
    st.plotly_chart(fig, use_container_width=True, key=f"temp_max_scatter_{key}")

def min_temp_scatter(width, height, key):
    fig = px.scatter(df, x='weather', y='temp_min', color='weather')
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Temperature (Celsius)',
        width=width,
        height=height
    )
    st.plotly_chart(fig, use_container_width=True, key=f"temp_min_scatter_{key}")

def plot_wind_scatter(width, height, key):
    fig = px.scatter(df, x='weather', y='wind', color='weather')
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Wind Speed (mph)',
        width=width,
        height=height
    )
    st.plotly_chart(fig, use_container_width=True, key=f"wind_speed_scatter_{key}")

def precipitation_scatter(width,height,key):
    fig = px.scatter(df, x='weather', y='precipitation', color='weather')
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Precipitation (inches)',
        width=width,
        height=height
    )
    st.plotly_chart(fig, use_container_width=True, key=f"precipitation_scatter_{key}")


def confusion_matrix():
    with open('./resource/confusion_matrix.pkl', 'rb') as f:
        cm = pickle.load(f)

    plt.figure(figsize=(1, 1))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['drizzle','fog','rain','snow','sun']).plot(cmap=plt.cm.Blues)
    plt.title('Weather Prediction Confusion Matrix')
    plt.xlabel('Predicted Weather')
    plt.ylabel('Actual Weather')

    st.pyplot(plt, use_container_width=True)

def resampled_pie(x, y):
    colors = ['skyblue', 'yellow', 'lightgreen', 'salmon', 'orange']
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(x, labels=y, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Weather Distribution After Borderline SMOTE')
    st.pyplot(fig)
########################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown("""
    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict weather conditions from the Seattle Weather Prediction dataset. The application uses Random Forest Classifier to forecast different weather conditions such as drizzle, rain, sun, snow, and fog.

    ### Pages

    - `Dataset` - Brief description of the Seattle Weather Prediction dataset used in this dashboard, including features like precipitation, temperature, and wind.

    - `EDA`- Exploratory Data Analysis of the Seattle Weather Prediction dataset. This section provides insights into the distribution of weather types and the relationships between weather features. It includes visualizations such as Pie Charts and Line Graphs.

    - `Data Cleaning / Pre-processing` - Data cleaning and preprocessing steps, including handling missing values, encoding weather types, and splitting the dataset into training and testing sets.

    - `Machine Learning`- Training supervised classification models, using **Random Forest Classifier**. This section also includes model evaluation and feature importance analysis to identify the key factors influencing weather predictions.

    - `Prediction` - An interactive page where users can input values for features such as temperature, precipitation, and wind to predict the weather condition using the trained models.

    - `Conclusion` - Summary of the key insights and observations from the EDA and model training phases. This section also discusses the strengths and limitations of the models and potential areas for further improvement.

    """)

   

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("WEATHER PREDICTION")
    st.markdown("""
    The **Seattle Weather Dataset** provides historical data on weather conditions in Seattle. The dataset was originally published on Kaggle by [Ananth](https://www.kaggle.com/ananthr1), with the goal of enabling weather pattern analysis and prediction. It includes measurements for various weather attributes, providing insight into Seattle's diverse weather conditions.

    This dataset is widely used in machine learning for classification tasks, as it contains labeled weather conditions such as drizzle, rain, sun, snow, and fog. For each observation, features such as precipitation, maximum and minimum temperature, and wind speed are recorded.

    ### Content
    The dataset consists of multiple rows, each representing a daily weather observation in Seattle. The primary attributes in the dataset are as follows:
    - **Precipitation**: The amount of precipitation recorded (in inches).
    - **Temp_Max**: Maximum daily temperature (in Fahrenheit).
    - **Temp_Min**: Minimum daily temperature (in Fahrenheit).
    - **Wind**: Wind speed recorded (in miles per hour).
    - **Weather**: Target variable indicating the type of weather observed (e.g., drizzle, rain, sun, snow, fog).

    Link: [Seattle Weather Dataset on Kaggle](https://www.kaggle.com/datasets/ananthr1/weather-prediction)

    ### Dataset displayed as a Data Frame
    """)
    st.dataframe(df.head())
    # Descriptive Statistics
    st.markdown("""
    ### Descriptive Statistics
    Below are some key descriptive statistics of the dataset to provide an overview of the variability and distribution of each feature. 
    """)
    st.write(df.describe())

    st.markdown("""
    The results from `df.describe()` provide key descriptive statistics for the Seattle weather dataset. First, the **precipitation** averages around {:.2f} inches with a standard deviation of {:.2f}, indicating some variation in rainfall levels throughout the year. **Maximum temperature**, meanwhile, has a mean of {:.2f} °F, with moderate variation around this average, while **minimum temperature** averages {:.2f} °F, suggesting consistent cooler ranges across seasons.

    When it comes to **wind speed**, the mean is approximately {:.2f} mph, showing a relatively steady distribution, though occasional spikes hint at windy days.

    Looking at minimum and maximum values, **precipitation** ranges from 0 up to higher levels during rainy days, and **temperature** varies significantly, reflecting the effects of seasonal change. Wind speed also exhibits some variability, indicating periods of high wind activity.

    The 25th, 50th, and 75th percentiles reveal a gradual increase across these features, demonstrating that this dataset captures a wide range of weather variables, which makes it well-suited for predictive modeling in weather classification.
    """.format(
    df["precipitation"].mean(),
    df["precipitation"].std(),
    df["temp_max"].mean(),
    df["temp_min"].mean(),
    df["wind"].mean()
))


# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    tab1, tab2 = st.tabs(["Graphs", "Insights"])

    # Your content for the EDA page goes here

    with tab1:
        col = st.columns((4, 3), gap='medium')
        col2 = st.columns((1,1,1,1), gap='small')

        with col[0]:
            st.markdown('#### Average Temperature Over Time')
            plot_average_temperature(300,200,1)

            st.markdown('#### Precipitation Over Time')
            plot_precipitation(300,200,1)

            st.markdown('#### Wind Speed Over Time')
            plot_wind(300,200,1)


        with col[1]:
            with st.expander('Legend', expanded=True):
                st.write('''
                    - Data: [Weather Prediction Dataset](https://www.kaggle.com/datasets/ananthr1/weather-prediction).
                    - :red[**Pie Chart**]: Distribution of weather conditions in the dataset.
                    - :blue[**Line Chart**]: Weather features over time.
                    - :green[**Scatterplot**]: Highlighting *overlaps* and *differences* between weather conditions.
                    ''')

            st.markdown('#### Weather Occurences in Seattle 2012-2015')
            weather_occurrences(450, 450, 1)

        with col2[0]:
            st.markdown('#### Maximum Temperature Distribution')
            max_temp_scatter(300,300,1)
        with col2[1]:
            st.markdown('#### Minimum Temperature Distribution')
            min_temp_scatter(300,300,1)
        with col2[2]:
            st.markdown('#### Precipitation Distribution')
            precipitation_scatter(300,300,1)
        with col2[3]:
            st.markdown('#### Wind Speed Distribution')
            plot_wind_scatter(300,300,1)    

    
    with tab2:
        st.markdown('#### Weather Occurences in Seattle 2012-2015')
        weather_occurrences(600, 600, 2)
        st.write("""
                Both rainy and sunny weather have the most 
                occurences in Seattle from 2012 to 2015 
                having 43.9% and 43.8% of the total 
                weather occurences, respectively.
                """)

        st.markdown('#### Average Temperature Over Time')
        plot_average_temperature(400,250,2)
        st.write("""
                The chart shows an identical weather 
                pattern for the city of Seattle, 
                Washington that has repeated each year 
                between 2012 to 2015. Where at the start 
                and near the end of every year, there is 
                significant drop in temparatures, ranging 
                from 0 to 10 degrees celsius, with some instances 
                in late 2013 and early 2014 almost reaching -5 
                degrees celsius. While, during the middle of the 
                year, we see the temparatures peaking between 20 
                to 25 degrees. This consistent pattern highlights 
                the clear seasonal climate variations experienced 
                by Seattle, with cold winters and warm summers repeating 
                predictably each year.
                """)

        st.markdown('#### Precipitation Over Time')
        plot_precipitation(400,250,2)
        st.write("""
                The precipitation trends for Seattle, Washington, 
                from 2012 to 2015 are depicted in the graph. 
                Throughout the period, precipitation levels 
                typically range from 0 to 55 inches. There is 
                clear variability in precipitation amounts, with
                notable rainfall events happening in 2013 and
                early 2015, despite the lack of noticeable seasonal
                peaks. Although there are no significant seasonal
                variations in rainfall over the years under
                observation, this steady pattern suggests that
                precipitation levels in Seattle are fluctuating.
                The data points to a generally steady but erratic
                precipitation trend, which is typical of Seattle's
                climate.
                """)

        st.markdown('#### Wind Speed Over Time')
        plot_wind(400,250,2)
        st.write("""
                The chart shows an identical wind pattern for the city of Seattle, Washington, 
                that has repeated each year between 2012 to 2015. Wind speeds generally fluctuate between 0 
                and 8 meters per second throughout the period. Notably, there are no distinct seasonal peaks, 
                but there is noticeable variability in wind speeds over time, with some periods experiencing higher gusts, 
                particularly in 2013 and late 2015. This consistent pattern suggests that while Seattle's wind conditions 
                are variable, there are no major seasonal extremes in wind speed over the observed years.
                """)

        st.markdown('#### Maximum Temperature Distribution')
        max_temp_scatter(450,300,2)
        st.write("""
                The distribution of the highest temperatures over different weather conditions has some observable patterns. 
                Sunny days have the highest maxima within the range of -1.6°C to 35.0°C, averaging about 19.9°C,
                which infers that on average, sunny conditions generally tend to be warmer overall. Rain days, 
                similarly, range quite widely from 3.9 to 35.6°C maximum temperatures, though average about 13.5°C, 
                which once more illustrates that rain can occur on cool as well as warm days. The maximum ranges for
                drizzle stand between 1.1°C and 31.7°C, whereas for fog, it lies between 1.7°C to 30.6°C. The average
                maximums of the two conditions are 15.9°C and 16.8°C, respectively, pointing towards moderate levels
                for both conditions but very occasionally warm. The coldest snowy days get with a maximum temperature
                only up to 11.1°C and an average of 5.6°C confirm that snow occurs exclusively in colder conditions.
                These trends indicate the predisposition for warmer maximum temperatures on sunnier and wetter days but
                for snow, which is limited to much cooler days.
                """)

        st.markdown('#### Minimum Temperature Distribution')
        min_temp_scatter(450,300,2)
        st.write("""
                The distributions of minimum temperatures under various weather 
                conditions are quite different. Drizzle and rain have similar 
                distributions of minimum temperatures, going typically from as low 
                as -3.9°C to as high as 18.3°C, which indicates generally mild to moderate minimum 
                temperatures with a significant overlap between the two distributions. The minimum 
                temperatures on sunny days tend to have the largest spread, ranging from -7.1°C to 18.3°C, 
                which is quite large. As shown in the diagram, minimum temperatures are the coldest on the 
                days with snow, ranging from -4.3°C up to 5.6°C, meaning that snow occurs only on the 
                coldest days. Foggy conditions relate to a rather wide range of minimum temperatures 
                between -3.2°C up to 17.8°C, meaning fog can occur in a colder or warmer environment. 
                These patterns reveal a broad variability in minimum temperatures for each of the weather types,
                with particularly a large range on sunny days that also corresponds to some of the lowest and some
                of the highest minimum temperatures in the dataset.
                """)

        st.markdown('#### Precipitation Distribution')
        precipitation_scatter(450,300,2)
        st.write("""
                The scatter plot summarises the rainfall and snowfall amounts
                at Seattle at different conditions. The highest range is for
                rain, which is from 50 inches. The strength of rain in Seattle
                \ varies very much, and such a large range is an indicator of it.
                Snow contains some difference in the amount of precipitation but
                the variance is not high for it. Drizzle and fog have almost no
                measurable precipitation, with very little outlier so that their
                precipitation is always light. Sun has no measurable precipitation,
                as one would expect. The amount of difference in precipitation between
                rain and the other weather conditions serves to show just how much more
                likely it is to rain than other weather conditions during large precipitation events in Seattle.
                """)

        st.markdown('#### Wind Speed Distribution')
        plot_wind_scatter(450,300,2)    
        st.write("""
                This scatter plot presents the distribution of wind speeds by weather type. Rainy conditions 
                clearly exhibit the highest spread, but there are many outliers that exceed 8 mph 
                by a significant amount; that is, rain often blows quite hard in Seattle. For snow,
                the pattern is similar, but the median for snow is much higher than for all other
                weather types. Interestingly, sunny weather conditions are associated with less wind
                speed, while drizzle and fog both show very little variability, including consistently
                low wind speeds. The general trend will be that higher wind speeds are generally linked
                to precipitation events, whereas calmer winds are typical for dry weather conditions such as sun, drizzle, and fog.
                """)



# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Display initial dataset overview
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # Displaying the initial data overview
    st.markdown("""
    Since our Seattle weather dataset is free of missing values and do not have any duplicate values it is all set to go with required data encoding and train/test split for our machine learning model. The columns of this dataset include:

    - date: Date of observation
    - precipitation: Precipitation in inches
    - temp_max and temp_min: Maximum and minimum temperatures in Celsius
    - wind: Average wind speed in mph
    - weather: Categorical weather condition (e.g., drizzle, fog, rain, snow, sun)
    """)

    # Display a sample of the dataset
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # Encoding the Weather Condition
    st.subheader("Encoding Weather Condition")
    st.markdown("""
    We used the LabelEncoder to convert our categorical column of weather to numbers. We used the column called weather_encoded as our label while training with mappings as given below:
    """)

    # Encode the weather column
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])

    # Display the encoding mappings
    weather_mapping_df = pd.DataFrame({'Weather Condition': le_weather.classes_, 'Encoded Value': range(len(le_weather.classes_))})
    st.dataframe(weather_mapping_df, use_container_width=True, hide_index=True)

    # Adding an Average Temperature Column
    st.subheader("Calculating Average Temperature")
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    st.markdown("""
     We added a new column temp_avg to indicate **average daily temperature** based on the two columns, temp_max and temp_min.
    """)

    st.dataframe(df[['temp_max', 'temp_min', 'temp_avg']].head(), use_container_width=True, hide_index=True)

    # Train-Test Split
    st.subheader("Train-Test Split")
    st.markdown("""
    The selected features for model training are:
    - **precipitation**: Amount of rainfall (in inches)
    - **temp_max** and **temp_min**: Maximum and minimum temperatures (in Celsius)
    - **wind**: Wind speed (in mph)
    
    The label is the **encoded weather condition** (weather_encoded).
    """)

    # Select features and target variable
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
    y = df['weather_encoded']

    st.code("""
    # Select features and target variable
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
    y = df['weather_encoded']
    """)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.code("""
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    """)

    # Display the training and testing data
    st.subheader("Training and Testing Data")
    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)

    st.markdown("We then split our dataset into training and test set and subsequently apply **oversampling** to address class imbalance.")

    # Applying Borderline SMOTE for Oversampling
    st.code("""
            # Applying Borderline SMOTE to deal with imbalanced data
            sampler = BorderlineSMOTE(random_state=42, sampling_strategy='auto', kind='borderline-2') 
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            """)

    sampler = BorderlineSMOTE(random_state=42, sampling_strategy='auto', kind='borderline-2')
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    st.markdown("""
    - Using **Borderline SMOTE** oversampling technique, class distribution was further balanced in training data (like for rarer conditions such as snow).
    - This also ensures that the model won't bias towards the more frequently appearing classes.
    """)

    # Calculate counts before SMOTE
    before_smote_counts = df['weather'].value_counts().sort_index()

    # Calculate counts after SMOTE
    after_smote_counts = pd.Series(Counter(y_train)).sort_index()
    after_smote_counts.index = le_weather.inverse_transform(after_smote_counts.index)  # Convert encoded labels back to original labels

    # Combine into a single DataFrame
    smote_comparison_df = pd.DataFrame({
        'Weather': before_smote_counts.index,
        'Before SMOTE': before_smote_counts.values,
        'After SMOTE': after_smote_counts.values
    })

    # Display the combined DataFrame
    st.subheader("Weather Distribution Before and After SMOTE")
    st.dataframe(smote_comparison_df, use_container_width=True, hide_index=True)

    # Visualization of Resampled Data
    resampled_pie(after_smote_counts.values, after_smote_counts.index)
    
    st.markdown("""
    Resampling would ensure every weather condition had adequate presentation, and therefore the model would then improve.
    """)

    st.markdown("We now train our supervised models on this balanced dataset after pre-processing is done.")

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
    st.write("""
            The confusion matrix reveals much deeper insight into how our model 
             is working after the balance with borderline SMOTE oversampling. It 
             says that the model is highly accurate on the overall predictions of 
             frequent weather conditions, that is mainly Rain and Sun, with 174 and 
             163 correct predictions respectively. It therefore means that the model 
             handles these major types of weather better. However after smoothening 
             with oversampling, the weather categories of Drizzle and Fog keep on being 
             frequently misclassified even today. Even with a relatively balanced dataset, 
             \some of the subtleties in the data are really hard to pick up by this model-for 
             instance, classifying Drizzle as Sun 12 times and classifying Fog as Sun 24 times. 
             Another class, Snow, happens by definition fewer times, and so prediction success is 
             also much lower, probably due to its different but less frequent occurrences in the data. 
             This performance suggests that, although the model is strong in common weather conditions 
             such as Rain and Sun, there is still a need for more fine-tuning, perhaps through further 
             feature engineering or model amendments to get better accuracy for edge case conditions 
             like Drizzle, Fog, and Snow, hence better generalization across all types of weathers.
            """)

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    
    
    # Units in the user input section
    precipitation = st.slider('Precipitation (in)', min_value=0.0, max_value=10.0, value=1.0)
    temp_max = st.slider('Max Temperature (°C)', min_value=-20, max_value=50, value=25)
    temp_min = st.slider('Min Temperature (°C)', min_value=-20, max_value=50, value=15)
    wind = st.slider('Wind Speed (mph)', min_value=0, max_value=150, value=10)

    # Create a DataFrame from user inputs
    user_input = pd.DataFrame({
        'precipitation': [precipitation],
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'wind': [wind]
    })


    # Predict the weather using the model
    prediction_encoded = model.predict(user_input)
    
    # Decode the predicted label
    le_weather = LabelEncoder()
    le_weather.fit(['drizzle', 'fog', 'rain', 'snow', 'sun'])  
    predicted_weather = le_weather.inverse_transform(prediction_encoded)[0]

    # Map the predicted weather condition to its corresponding emoji
    if predicted_weather == "rain":
        emoji = "🌧️"
    elif predicted_weather == "sun":
        emoji = "☀️"
    elif predicted_weather == "snow":
        emoji = "❄️"
    elif predicted_weather == "fog":
        emoji = "🌫️"
    else:
        emoji = "🌦️"

    # Show the predicted weather with corresponding emoji
    st.write(f"Predicted weather: {emoji} {predicted_weather}")

# Conclusions Page
if st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
    Through exploratory data analysis and the application of machine learning techniques to weather data, several key insights and observations have emerged:

    #### 1. 📊 **Weather Occurrences**:
    - Rainy and sunny weather types were the most frequent, dominating the observed period in Seattle. These weather patterns, due to their frequency, offer a reliable foundation for predictive models.

    #### 2. 📈 **Temperature Trends**:
    - A clear seasonal pattern is present in the temperature data. Temperatures tend to drop significantly at the beginning and end of each year, coinciding with colder winters, while peaking in the middle, aligning with warmer summers.

    #### 3. 🌧️ **Precipitation Trends**:
    - Precipitation data between 2012 and 2015 reveals significant variability, ranging from 0 to 55 inches. High precipitation levels were generally tied to rainy or snowy weather, with a more consistent correlation to rain.

    #### 4. 💨 **Wind Speed Trends**:
    - Wind speeds fluctuated between 0 and 8 meters per second throughout the observation period. Windy conditions were more commonly seen during rainy or snowy weather, while calmer winds were observed in drizzle, sunny, and foggy weather.

    #### 5. 🤖 **Machine Learning Model Insights**:
    - A Borderline-SMOTE technique was used to handle imbalanced data, oversampling minority features to improve model balance.
    - The **Random Forest Classifier** achieved an accuracy of around 80%. The model performed well in predicting sunny and rainy weather, though it struggled with drizzle, snow, and fog conditions. This was evident from the confusion matrix, which highlighted the model's lower accuracy for these weather types.
    - Further data on drizzle, snow, and fog conditions would likely improve prediction accuracy and allow the model to better handle the full spectrum of weather types.

    #### **Summing up**:
    The weather data analysis revealed key trends, with clear seasonal patterns and relationships between weather occurrences, temperature, precipitation, and wind speed. The machine learning model showed promising accuracy for more common weather types but needs more data for less frequent conditions to enhance prediction performance. With further refinement and data collection, the model could achieve even better accuracy across all weather categories.
    """)

    