# Seattle Weather Prediction Dashboard

A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict weather conditions from the Seattle Weather Prediction dataset. The application uses Random Forest Classifier to forecast different weather conditions such as drizzle, rain, sun, snow, and fog.

![Main Page Screenshot](./resource/image/main_page.png)

### 👥 Group Members:
1. Butial, Aivann Paul O.
2. Lim, Evan Vincent B.
3. Lim, Kyle Hendrik L.
4. Ongtangco, Randolph Joshua M.
5. Tan, Gabriel Christian D.

### 🔗 Links:

- 🌐 [Streamlit Link](https://seattle-weather-prediction.streamlit.app/)
- 📗 [Google Colab Notebook](https://colab.research.google.com/drive/1xwaCdEhWPi_2sUwpqr9Mp2uCdcOyj45_#scrollTo=MQo1i9FqwRfd)

### 📊 Dataset:

- [Weather Prediction (Kaggle)](https://www.kaggle.com/datasets/ananthr1/weather-prediction)

### 📖 Pages:

1. `Dataset` - Brief description of the Seattle Weather Prediction dataset used in this dashboard, including features like precipitation, temperature, and wind.
2. `EDA`- Exploratory Data Analysis of the Seattle Weather Prediction dataset. This section provides insights into the distribution of weather types and the relationships between weather features. It includes visualizations such as Pie Charts and Line Graphs.
3. `Data Cleaning / Pre-processing` - Data cleaning and preprocessing steps, including handling missing values, encoding weather types, and splitting the dataset into training and testing sets.
4. `Machine Learning`- Training supervised classification models, using **Random Forest Classifier**. This section also includes model evaluation and feature importance analysis to identify the key factors influencing weather predictions.
5. `Prediction` - An interactive page where users can input values for features such as temperature, precipitation, and wind to predict the weather condition using the trained models.
6. `Conclusion` - Summary of the key insights and observations from the EDA and model training phases. This section also discusses the strengths and limitations of the models and potential areas for further improvement.

### 💡 Findings / Insights

Through exploratory data analysis and training `Random Forest Classifier` on the **Weather Prediction Dataset**, the key insights and observations are:

#### 1. 📊 **Dataset Characteristics**:

- The dataset shows that, among the weather elements (features), `temp_max` has the highest variability with a standard deviation of **7.35**. `wind`, however, has the lowest variability with a standard deviation of **1.44**.
- The dataset's `weather` distribution are significantly imbalanced, wherein `rain` and `sun` combined makeup about **88%** of the distribution. An oversampling method, Borderline Smote 2, was used to handle this data imbalance.
- The dataset has no null values and has no inconsistent data formatting.

#### 2. 📝 **Feature Distributions and Separability**:

- **Scatter Plot** analysis indicates that most weather conditions are overlapping across all weather elements or features. Overlapping features may negatively affect the prediction accuracy of the model. 
- **Precipitation** emerged as the most discriminative feature especially for distinguishing `rain` and `snow` weather conditions, but more so on the former at high amounts of precipitaion.

#### 3. 📈 **Model Performance (Random Forest Classifier)**:

- The `Random Forest Classifier` achieved about 80% accuracy on the training data considering imbalanced weather distribution regardless of oversampling the minority using Borderline SMOTE 2. The model's prediction on `rain` and `sun` weather conditions have the highest accuracy.
- In terms of **feature importance** results from the trained model indicate that `temp_max` followed by `precipitation` were the dominant predictors having **29%** and **27%** importance values, respectively.

##### **Summary:**

The weather data analysis revealed key trends, with clear seasonal patterns and relationships between weather occurrences, temperature, precipitation, and wind speed. The machine learning model showed promising accuracy for more common weather types but needs more data for less frequent conditions to enhance prediction performance. With further refinement and data collection, the model could achieve even better accuracy across all weather categories.