from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            job=request.form.get('job'),
            loan = request.form.get('loan'),
            Gender = request.form.get('gender'),
            marital=request.form.get('marital'),
            education=request.form.get('education'),
            poutcome=request.form.get('poutcome'),
            balance=int(request.form.get('balance')),
            duration=int(request.form.get('duration')),
            housing = request.form.get('housing'),
            campaign = int(request.form.get('campaign')),
            Count_Txn=int(request.form.get('Count_Txn')),
            Annual_Income=int(request.form.get('Annual_Income')),
            age = int(request.form.get('age'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


