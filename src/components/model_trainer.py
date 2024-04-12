import os
import sys
import mlflow
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from mlpro.exception import CustomException
from mlpro.logger import logging

from mlpro.utils import save_object,evaluate_models

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
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier()
            }
            params = {
            "Logistic Regression": {
                'C': [0.1, 1.0]
            },

            "Decision Tree": {
                'criterion': ['gini', 'entropy'],  # Different criterion for classification   # Splitting strategy
                'max_depth': [None, 3, 5, 7]    
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256],           # Criterion for splitting
                'max_depth': [None, 3, 5, 7]   
            },
            "XGBClassifier": {
                'learning_rate': [0.1, 0.01],  # Learning rate (eta) for boosting
                'n_estimators': [8, 16, 32],  # Number of boosting rounds
                'max_depth': [3, 5, 7]              # Minimum sum of instance weight needed in a child
            }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)

            ## To get best model score from dict
            best_model_name = max(model_report, key=lambda x: model_report[x]['recall'])
            best_model = models[best_model_name]
            accuracy = model_report[best_model_name]["accuracy"]
            

            logging.info(f"Best found model on both training and testing dataset - {best_model} with {model_report[best_model_name]['accuracy']}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Return accuracy
            return accuracy
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
        
            raise CustomException(e,sys)

