import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
import mlflow
experiment_name = "Multiple_Models_Experiment"
mlflow.set_tracking_uri("mlruns")
# Start the MLflow experiment
mlflow.set_experiment(experiment_name)
from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
from sklearn.model_selection import GridSearchCV
from .logger import logging
from .exception import CustomException


    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for name, model in models.items():
            with mlflow.start_run():
            
                logging.info(f"Starting the model: {name}")
                params = param.get(name, {})  # Get parameters for the current model
                mlflow.set_tag("model",f"{name}")

                gs = GridSearchCV(model, params, cv=5)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                best_model.fit(X_train, y_train)
                y_test_pred = best_model.predict(X_test)
                logging.info(f"Model trained: {name}")
                acc = accuracy_score(y_test, y_test_pred)
                prec = precision_score(y_test, y_test_pred)
                recall = recall_score(y_test, y_test_pred)
                entropy = log_loss(y_test, y_test_pred)
                mlflow.log_params(gs.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("entropy", entropy)
                mlflow.sklearn.autolog()

                model_report = {
                    'accuracy': round(acc, 2),
                    'precision': round(prec, 2),
                    'recall': round(recall, 2),
                    'entropy': round(entropy, 2)
                }
                report[name] = model_report
            print(report)
            logging.info("Model evaluation completed")
        mlflow.end_run()

        return report
    except Exception as e:
        raise CustomException(e, sys)
    

    
def load_object(file_path):
    try:

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)