import numpy as np
import config
import preprocessing as pp
import warnings
import joblib

# =========================================
# Model Training
# 1. Load The Dataset

df = pp.load_data(config.PATH_TO_DATASET)

# 2. Divide the Data set

x_train, x_test, y_train, y_test = pp.splitting_data(df, config.TARGET)

# 3. Impute Categorical Variables
x_train[config.CATEGORICAL_TO_IMPUTE] = pp.impute_na(x_train, config.CATEGORICAL_TO_IMPUTE)