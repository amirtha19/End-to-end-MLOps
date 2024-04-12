# End-to-end Machine Learning Project

This project demonstrates an end-to-end machine learning pipeline, including data ingestion and a machine learning application.

## Setup

1. **Clone the repository:**
    ```bash
    git clone 
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the MLflow backend server:**
    ```bash
    mlflow server --backend-store-uri mlruns --default-artifact-root mlruns
    ```

4. **Start the MLflow artifact server** (in another command prompt or terminal window):
    ```bash
    mlflow artifacts serve --backend-store-uri mlruns --default-artifact-root mlruns
    ```

## Usage

1. **Run the data ingestion script:**
    ```bash
    python data_ingestion.py
    ```

2. **Train the model and start the application:**
    ```bash
    python app.py
    ```

3. **Access the MLflow UI** to track experiments and manage models:
    Open a web browser and go to `http://localhost:5000` for the MLflow backend server or `http://localhost:8000` for the MLflow artifact server.

## Components

- `data_ingestion.py`: Script for ingesting and preprocessing data.
- `app.py`: Main application script for training models and serving predictions.
- `model_evaluation.py`: Script for evaluating models and selecting the best one.

## Dependencies
`requirements.txt`
