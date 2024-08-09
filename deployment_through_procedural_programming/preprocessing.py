import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import joblib

# ===========================================================================
# Individual preprocessing and Training functions

# 1. Loading the data


def load_data(path):
    return pd.read_csv(path)

# 2. Dividing the Data Set in to Training and Testing Sets


def splitting_data(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data.drop(target,
                                                                  axis=1),
                                                        data[target],
                                                        test_size=0.1,
                                                        random_state=0
                                                        )
    return x_train, x_test, y_train, y_test

# 3. Replacing categorical missing values


def impute_na(data, var, replacement="Missing"):
    return data[var].fillna(replacement)

# 4. Capture elapsed year values


def elapsed_year(data, var, ref_var="YrSold"):
    return (data[ref_var] - data[var])

# 5. Log transform of numerical variables


def log_transform(data, var):
    return np.log(data[var])

# 6. Removing Rare Labels


def rare_labels(data, var, frequent_ls):
    data[var] = np.where(data[var].isin(frequent_ls), data[var], "Rare")
    return data[var]

# 7. Encode Categorical Variables


def encode_cat(data, var, mappings):
    data[var] = data[var].map(mappings)
    return data[var]

# 8. fitting the min max scaler and saving the scaler


def train_scaler(data, output_path):
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, output_path)
    return scaler

# 9. scale the features


def scale_features(data, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(data)

# 10. model training and saving


def train_model(data, target, model_path):
    model = Lasso(alpha=0.001, random_state=0)
    model.fit(data, target)
    joblib.dump(model, model_path)
    return None

# making Predictions using the model


def predictions(data, model_path):
    model = joblib.load(model_path)
    pred = model.predict(data)
    return pred
