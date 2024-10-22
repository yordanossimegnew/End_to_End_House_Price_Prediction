# End_to_End_House_Price_Prediction

![House Prices](https://github.com/yordanossimegnew/End_to_End_House_Price_Prediction/blob/690f40979a41340c5f56a59e7e7baaef8265c474/house%20image.jpg)
![Distribution of Residuals](https://github.com/yordanossimegnew/End_to_End_House_Price_Prediction/blob/690f40979a41340c5f56a59e7e7baaef8265c474/reports/figures/Distibution%20of%20Residuals.jpg)
![Actual vs Predicted Sale Price](https://github.com/yordanossimegnew/End_to_End_House_Price_Prediction/blob/690f40979a41340c5f56a59e7e7baaef8265c474/reports/figures/Actual%20Vs%20Pridicted%20Sale%20Price.jpg)

## Overview
Welcome to the House Price Prediction project! This repository contains a comprehensive machine learning pipeline designed to predict house prices using Lasso regression technique. My goal is to leverage data science to provide accurate and reliable predictions, enabling better decision-making in the real estate market.

## Features
- **Data Preprocessing**: Clean and prepare raw data for analysis.
- **Feature Engineering**: Transform and engineer features to enhance model performance.
- **Feature Selection**: Selecting good features for the model prediction.
- **Model Training**: Implement and train regression models.
- **Model Evaluation**: Evaluate model performance using r2_score and mean_squared_error.
- **Model Deployment**: Serialize and deploy models for real-time prediction.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Model Deployment](#model-deployment)
9. [Contributing](#contributing)

## Installation
To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yordanossimegnew/End_to_End_House_Price_Prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

## Usage
Follow the steps below to run the project:

1. **Load Data**: Place your dataset in the `data/raw_data` directory.
2. **Run Preprocessing**: Execute the preprocessing script to clean and prepare the data.
3. **Train Model**: Train the model using the prepared data.
4. **Evaluate Model**: Assess the model's performance.
5. **Make Predictions**: Use the trained model to make predictions on new data.

```python
python preprocess.py
python train.py
python evaluate.py
python predict.py --input data/raw_data/test.csv
```

## Project Structure
```
house-price-prediction/
├── data/
│   ├── raw_data/
│   └── processed_data/
├── models/
│   ├── scaler.pkl
│   └── lasso_regression.pkl
├── notebooks/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── reports
├── README.md
└── requirements.txt
```

## Data Preprocessing
Preprocessing steps include handling missing values, feature transformation, and encoding categorical variables. These steps ensure that the data is clean and ready for model training.

```python
def load_data(path):
    return pd.read_csv(path)

def preprocess_data(data):
    # Implement preprocessing steps
    return processed_data
```

## Feature Engineering
Feature engineering involves creating new features and transforming existing ones to improve model performance. Techniques used include log transformation, encoding categorical variables, and binarize skewed features.

```python
def feature_engineering(data):
    # Implement feature engineering steps
    return engineered_data
```

## Model Training
Train the Lasso regression model using the preprocessed data. The model is saved for future predictions.

```python
def train_model(data, target):
    model = Lasso(alpha=0.001, random_state=0)
    model.fit(data, target)
    joblib.dump(model, 'models/lasso_regression.pkl')
```

## Model Evaluation
Evaluate the model using metrics such as Mean Squared Error (MSE) and R-Squared (R2). These metrics help in understanding the model's performance.

```python
def evaluate_model(model, test_data, test_target):
    predictions = model.predict(test_data)
    mse = mean_squared_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)
    return mse, r2
```

## Model Deployment
Deploy the trained model for making real-time predictions. The model and scaler are loaded to transform new data and make predictions.

```python
def predict(data):
    model = joblib.load('models/lasso_regression.pkl')
    predictions = model.predict(data)
    return predictions
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
