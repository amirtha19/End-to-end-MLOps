import threading
import mlflow
from mlflow import cli
server_thread = threading.Thread(target=mlflow.cli.server())
server_thread.start()
print("What")
