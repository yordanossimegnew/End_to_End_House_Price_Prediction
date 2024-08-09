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

# 3. Impute categorical missing values
for var in config.CATEGORICAL_TO_IMPUTE:
    x_train[var] = pp.impute_na(x_train, var)

# 4. Impute Numerical Variables
x_train[config.NUMERICAL_TO_IMPUTE] = pp.impute_na(x_train, config.NUMERICAL_TO_IMPUTE, config.LOTFRONTAGE_MODE)

# 5. Captured elapsed time
x_train[config.YEAR_VARIABLE] = pp.elapsed_year(x_train, config.YEAR_VARIABLE)

# 6. Log transform numerical variables
for var in config.NUMERICAL_LOG:
    x_train[var] = pp.log_transform(x_train, var)
    
# 7. Group rare labels
for var in config.FREQUENT_LABELS:
    x_train[var] = pp.rare_labels(x_train, var, config.FREQUENT_LABELS[var])
    
# 8. Encoding categorical variables
for var in config.CATEGORICAL_ENCODE:
    x_train[var] = pp.encode_cat(x_train, var , config.ENCODING_MAPPINGS[var])
    
# 9. fitting the min max scaler and saving the scaler
scaler = pp.train_scaler(x_train[config.FEATURES],
                         config.OUTPUT_SCALER_PATH)

# 10. Sclae the train set
x_train = pp.scale_features(x_train[config.FEATURES],config.OUTPUT_SCALER_PATH)

# 11. Train and save the model
pp.train_model(x_train, np.log(y_train), config.OUTPUT_MODEL_PATH)

print("Finished Training")