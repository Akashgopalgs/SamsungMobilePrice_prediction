# Samsung Mobile Price Prediction

Welcome to the Samsung Mobile Price Prediction repository! This project involves predicting the prices of Samsung mobiles using machine learning techniques. The project includes Web-Scraping, data preprocessing, model training, and a Streamlit web application for price prediction.
![Screenshot 2024-08-11 000950](https://github.com/user-attachments/assets/9d5070a6-c70f-42b9-89db-32519bb33469)

## Project Overview

The goal of this project is to predict the price of Samsung mobile phones based on various features such as battery capacity, storage, RAM, rating, and camera specifications. The project involves:

1. **Data Collection**: Scraping mobile data from Flipkart.
2. **Data Preprocessing**: Cleaning and transforming the data.
3. **Exploratory Data Analysis (EDA)**: Analyzing data distributions and correlations.
4. **Model Training**: Building and evaluating machine learning models.
5. **Deployment**: Creating a Streamlit application for price prediction.

## Data Source

- **Data Source**: Crapped Samsung mobile features data from Flipkart.

## Files

- `samsung_Mobiles.json`: The dataset used for this project.
- `best_model.pkl`: The trained machine learning model.
- `encoder.pkl`: The encoder used for feature transformation.
- `scaler.pkl`: The scaler used for feature scaling.
- `app.py`: The Streamlit application for predicting mobile prices.


## Acknowledgements

- **Data Source**: Flipkart
- **Libraries**: Pandas, NumPy, scikit-learn, Streamlit, Matplotlib

## Streamlit App

You can access the deployed Streamlit application here: [Samsung Mobile Price Prediction](https://samsungmobilepriceprediction-jgqfwqzbdumxxpdezhqwcg.streamlit.app/)

## Model Training & Evaluation

The following models were evaluated for their performance:

- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor

**Decision Tree Regressor** was selected due to its high accuracy of 92%.

