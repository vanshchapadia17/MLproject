import os
import sys
from dataclasses import dataclass

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifects',"model.pkl")
    model_report_file_path=os.path.join('artifects',"model_report.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spli training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)

            report_df = pd.DataFrame.from_dict(model_report, orient="index")
            report_df = report_df.sort_values("test_r2", ascending=False)

            logging.info("===== Model training & comparison report (sorted by test R^2) =====")
            header = (
                f"{'Model':<25s} | {'train R2':>8s} | {'test R2':>8s} | "
                f"{'train MAE':>9s} | {'test MAE':>9s} | "
                f"{'train RMSE':>10s} | {'test RMSE':>10s} | {'fit (s)':>7s} | best_params"
            )
            logging.info(header)
            logging.info("-" * len(header))
            print(header)
            print("-" * len(header))
            for name, row in report_df.iterrows():
                line = (
                    f"{name:<25s} | {row['train_r2']:>8.4f} | {row['test_r2']:>8.4f} | "
                    f"{row['train_mae']:>9.3f} | {row['test_mae']:>9.3f} | "
                    f"{row['train_rmse']:>10.3f} | {row['test_rmse']:>10.3f} | "
                    f"{row['fit_time_s']:>7.2f} | {row['best_params']}"
                )
                logging.info(line)
                print(line)

            os.makedirs(os.path.dirname(self.model_trainer_config.model_report_file_path), exist_ok=True)
            report_df.to_csv(self.model_trainer_config.model_report_file_path, index_label="model")
            logging.info(f"Saved model comparison to {self.model_trainer_config.model_report_file_path}")

            best_model_name = report_df.index[0]
            best_model_score = float(report_df.iloc[0]["test_r2"])
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model: {best_model_name} (test R^2 = {best_model_score:.4f})")
            print(f"\nBest model: {best_model_name} (test R^2 = {best_model_score:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)

