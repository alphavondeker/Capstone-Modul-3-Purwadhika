# Bike Sharing Demand Prediction - Capstone 3


Gerard Louis Howan - JCDS-2502

Dataset : Bike Sharing
## Introduction

This project aims to predict the demand for bike rentals in a bike-sharing system using historical data. The model is built using the XGBoost Regressor algorithm and various preprocessing and feature engineering techniques.

## Contents

1. Business Problem Understanding
2. Data Understanding
3. Data Preprocessing
4. Modeling
5. Limitations
6. Conclusion
7. Recommendation

## 1. Business Problem Understanding

### Context

Bike-sharing systems are a new generation of traditional bike rentals where the whole process, from membership, rental, and return back, has become automatic. These systems generate data that can be used for research and to improve the efficiency of the bike-sharing service.

### Problem Statements

One of the challenges in bike-sharing systems is the allocation of bikes. The company must provide a sufficient number of bikes to meet user demands. A prediction model can help in maintaining the efficiency of operating costs and minimizing risks.

### Goals

The goal is to develop a model that predicts the number of bikes needed based on historical data and existing conditions.

### Analytic Approach

An exploratory data analysis (EDA) is performed to determine the features and their relation to the target outcome. A regression machine learning model is then defined, trained, and tested.

### Metric Evaluation

The following metrics are used to evaluate the model:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R-squared (R², Coefficient of Determination)

## 2. Data Understanding

The dataset contains records of bike rentals from 2011-2012. Each row represents information related to the time of rental, weather, and corresponding season.

### Attributes Information

| Attribute   | Data Type | Description                                                                 |
|-------------|-----------|-----------------------------------------------------------------------------|
| dteday      | Object    | Date                                                                        |
| hum         | Float     | Normalized humidity (values divided by 100)                                  |
| weathersit  | Integer   | Weather situation (1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Rain/Snow)|
| holiday     | Integer   | Holiday indicator (0: Not holiday, 1: Holiday)                               |
| season      | Integer   | Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)                            |
| atemp       | Float     | "Feels like" temperature in Celsius                                          |
| temp        | Float     | Normalized temperature in Celsius                                            |
| hr          | Integer   | Hour (0 to 23)                                                               |
| casual      | Integer   | Count of casual users                                                        |
| registered  | Integer   | Count of registered users                                                    |
| cnt         | Integer   | Count of total rental bikes including both casual and registered users       |

## 3. Data Preprocessing

### Steps

1. Convert `dteday` to datetime format and extract year, month, and day.
2. Map the `season` column to corresponding season names.
3. Convert categorical columns to category data type.
4. Drop irrelevant features (`registered`, `casual`, `atemp`, `year`).
5. Check for missing values and duplicates.
6. Handle outliers in numerical features.

## 4. Modeling

### Feature and Target Definition

The target variable (`cnt`) is log-transformed to stabilize variance and improve model accuracy.

### Train and Test Splitting

The data is split into 80% training and 20% testing sets.

### Encoding & Scaling

- OneHot Encoding: `weathersit`, `holiday`, `season`
- Binary Encoding: `month`, `day`
- RobustScaler: `hr`
- Polynomial Features: `temp`, `hum`, `hr`

### Benchmark Model

Various models are compared based on RMSE, MAE, MAPE, and R² metrics. The XGBoost Regressor is selected as the best model.

### Hyperparameter Tuning

RandomizedSearchCV is used to tune the hyperparameters of the XGBoost model.

## 5. Limitations

### Feature Limitations

For this model to be used,  data must follow these constraints:
| **Feature** | **Type** | **Acceptable Inputs** |
| --- | --- | --- |
| hum | Float | 0.0 to 1.0 |
| weathersit | Category | {'1', '2', '3', '4'} |
| holiday | Category | {'0', '1'} |
| season | Category | {'winter', 'spring', 'summer', 'fall'} |
| temp | Float | 0.0 to 1.0 |
| hr | Integer | 0 to 23 |
| month | Category | {'January','February',...,'December'} |
| day | Category | {'Monday','Tuesday',...,'Sunday'} |

Any input outside of these boundaries might cause inaccuracy or failure in the model

### Target Limitations

- The target used in this model is log-transformed (np.log1p(y)) before modeling. For example, if the target is 200 bikes, the target inputted to the model will be log1p(200)

- The prediction output of this model will also be in log scale and **must be exponentiated (np.expm1(y_pred)) to get actual values**

- Since the target is log-transformed, target values must not be a negative value

### Model Limitations

- Predictions below 400 are generally accurate.
- Predictions between 400 and 600 show higher variability.
- Predictions above 600 tend to be underestimated.

## 6. Conclusion

The XGBoost Regressor model with tuned hyperparameters performs well with a MAPE of 0.337 on actual values. The most influential features are hour, humidity, season, and weather conditions.

## 7. Recommendation

- Experiment with alternative techniques to handle skewness.
- Apply post-prediction adjustments for higher values.
- Introduce new features and more data to improve the model.
