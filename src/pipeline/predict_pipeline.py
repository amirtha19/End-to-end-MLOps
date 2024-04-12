import sys
import os
import pandas as pd
from src.components.mlpro.exception import CustomException
from src.components.mlpro.utils import load_object
import mlflow

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_uri = 'runs:/494f468a58b548da8e49cbedf9752c3d/model'
            model = mlflow.pyfunc.load_model(model_uri)
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
numerical_features = ['balance', 'duration', 'previous', 'Count_Txn', 'Annual_Income',"duration","campaign","age"]
categorical_features = ['job', 'marital', 'education', 'poutcome',"Insurance","housing","Gender"]
class CustomData:
    def __init__(  self,
        job: str,
        age:int,
        campaign:str,
        Gender:str,
        marital: str,
        housing :str,
        education:str,
        poutcome: str,
        loan:str,
        balance: int,
        duration:int,
        Count_Txn:int,
        Annual_Income:int):

        self.job = job
        self.age = age
        self.campaign = campaign
        self.marital = marital
        self.housing = housing
        self.education = education
        self.poutcome = poutcome
        self.loan = loan
        self.Gender = Gender
        self.balance = balance
        self.duration = duration
  
        self.Count_Txn = Count_Txn
        self.Annual_Income = Annual_Income

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "loan":[self.loan],
                "Gender":[self.Gender],
                "housing":[self.housing],
                "age" : [self.age],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "poutcome": [self.poutcome],
                "balance": [self.balance],
                "duration": [self.duration],
                "Count_Txn": [self.Count_Txn],
                "Annual_Income": [self.Annual_Income],
                "campaign": [self.campaign]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)