import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import accuracy_score
MLFLOW_TRACKING_URI = "mlruns"
mlflow.set_tracking_uri("mlruns")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
client.search_experiments()
experiments = client.search_experiments()
experiment_list = [exp for exp in experiments]
print(experiment_list)

def test_model(name, stage, X_test, y_test):
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred, squared=False)}

def find_best_runs(X_test, y_test):
    for experiment_id in experiment_list:
        runs = client.search_runs(
            experiment_ids=experiment_id,
            filter_string="metrics.accuracy < 85",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.recall DESC"]
        )

        for run in runs:
            print(f"run id: {run.info.run_id},run_name:{run.info.run_name} accuracy: {run.data.metrics['accuracy']:.4f}, recall: {run.data.metrics['recall']:.4f}")
            run_id = run.info.run_id
            run_name = run.info.run_name

            model_uri = f"mlruns:/{run_id}/model"
            latest_versions = client.get_latest_versions(name=run_name)

            for version in latest_versions:
                print(f"version: {version.version}, stage: {version.current_stage}")
                
                evaluation_result = test_model(run_name, version.current_stage, X_test, y_test)
                print(f"Evaluation result for model {run_name} version {version.version}: {evaluation_result}")

# Assuming X_test and y_test are already defined
find_best_runs(X_test, y_test)
