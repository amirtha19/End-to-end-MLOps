import mlflow.sklearn
from mlflow import MlflowClient
import sys
import os
from mlflow.entities import ViewType
from sklearn.metrics import accuracy_score
from mlpro.exception import CustomException
MLFLOW_TRACKING_URI = "mlruns"
mlflow.set_tracking_uri("mlruns")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
client.search_experiments()
experiments = client.search_experiments()
experiment_list = [exp for exp in experiments]
print(experiment_list)
import dill
import pickle
def test_model(name, stage, X_test, y_test):
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred, squared=False)}

def find_best_runs():
    for experiment in experiment_list:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(
            experiment_ids=experiment_id,
            filter_string="metrics.accuracy < 85",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.recall DESC"]
        )

        for run in runs:
            print(f"run id: {run.info.run_id}, run_name: {run.info.run_name}, accuracy: {run.data.metrics['accuracy']:.4f}, recall: {run.data.metrics['recall']:.4f}")
            run_id = run.info.run_id
            run_name = run.info.run_name

            model_uri = f"mlartifacts/{run_id}/best_estimator/model.pkl"
            print(model_uri)
        return model_uri

def save_object(file_path,obj):
    try:
        
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)             
    
