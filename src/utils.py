import os
import sys
import time
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for name, model in models.items():
            para = param[name]

            start = time.time()
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            fit_time = time.time() - start

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            report[name] = {
                "train_r2":   r2_score(y_train, y_train_pred),
                "test_r2":    r2_score(y_test, y_test_pred),
                "train_mae":  mean_absolute_error(y_train, y_train_pred),
                "test_mae":   mean_absolute_error(y_test, y_test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "test_rmse":  np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "fit_time_s": fit_time,
                "best_params": gs.best_params_,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)