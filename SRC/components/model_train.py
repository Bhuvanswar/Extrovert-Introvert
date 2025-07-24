import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as AS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..exception import CustomException
from ..logger import logging

from ..utility import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "LogisticRegression": LogisticRegression(),
                #"XGBClassifier": XGBClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],  # Correct values for classification
                    # 'splitter':['best', 'random'],  # Optional for tuning
                    # 'max_features':['sqrt', 'log2'],  # Optional for tuning
                },
                "Random Forest": {
                    'criterion':['gini', 'entropy'],  # Uncomment for classification
                    'max_features':['sqrt', 'log2', None],  # Optional for tuning
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],  # For regression
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LogisticRegression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'newton-cg', 'lbfgs'],
                    'max_iter': [100, 200, 300]
                },
                #"XGBRegressor": {
                #    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                #    'n_estimators': [8, 16, 32, 64, 128, 256]
                #},
                "KNeighborsClassifier":{
                    'n_neighbors': [3, 5, 7, 9, 11],        # Number of neighbors to use
                    'weights': ['uniform', 'distance'],    # Weight function used in prediction
                    'metric': ['euclidean', 'manhattan'] 
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(best_model_name)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            Acc = AS(y_test, predicted)
            logging.info(Acc)
            return AS
              
        except Exception as e:
            raise CustomException(e,sys)