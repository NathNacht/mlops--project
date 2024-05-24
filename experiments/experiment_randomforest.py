import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate, train_test_split
from mlflow.models.signature import infer_signature
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from utils.data_preprocessing import load_data, preprocess_data
from utils.model_utils import train_model, evaluate_model
from models.model_rf import get_model
from typing import Dict, Any

# Setting the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment to "Sklearn Model"
mlflow.set_experiment("Sklearn RandomForestClassifier")

# Load and preprocess data
filename = 'BankChurners.csv'
df = load_data(filename)
df = preprocess_data(df)

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into train and test sets

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing data and labels.
    """
    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def log_model_and_metrics(trained_model: Any, X_train: pd.DataFrame, y_train: pd.Series, evaluation_results: Dict[str, Any]) -> None:
    """
    Logs the trained model and evaluation metrics.

    Parameters:
    trained_model (Any): The trained machine learning model.
    X_train (pd.DataFrame): The features used for training the model.
    y_train (pd.Series): The target variable used for training the model.
    evaluation_results (Dict[str, Any]): A dictionary containing evaluation metrics.
    model_type (str): The type of the model.

    Returns:
    None
    """
    # Log the model
    model_signature = infer_signature(X_train, y_train, params={'model_type': 'RandomForest'})
    print(model_signature.to_dict())
    mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="random_forest_classifier_trained",
        conda_env=None,
        signature=model_signature,
        registered_model_name="random_forest_classifier_trained"
    )

    # Log metrics
    mlflow.log_metric('accuracy', evaluation_results['accuracy'])
    mlflow.log_metric('roc_auc', evaluation_results['roc_auc'])
    
    with open("metrics/classification_report.txt", "w") as f:
        f.write(evaluation_results['classification_report'])
    mlflow.log_artifact("metrics/classification_report.txt", artifact_path="metrics")

def save_and_log_plots(trained_model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Saves and logs ROC and Precision-Recall curves.

    Parameters:
    trained_model (Any): The trained machine learning model.
    X_test (pd.DataFrame): The features used for testing the model.
    y_test (pd.Series): The target variable used for testing the model.

    Returns:
    None
    """
    roc_display = RocCurveDisplay.from_estimator(trained_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig("metrics/roc_curve.png")
    plt.close()
    mlflow.log_artifact("metrics/roc_curve.png", artifact_path="metric_graphs")

    pr_display = PrecisionRecallDisplay.from_estimator(trained_model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("metrics/precision_recall_curve.png")
    plt.close()
    mlflow.log_artifact("metrics/precision_recall_curve.png", artifact_path="metric_graphs")

def cross_validate_model(trained_model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Performs cross-validation and prints mean accuracy, fit time, and score time.

    Parameters:
    trained_model (Any): The trained machine learning model.
    X_train (pd.DataFrame): The features used for training the model.
    y_train (pd.Series): The target variable used for training the model.

    Returns:
    None
    """
    cv_results = cross_validate(trained_model, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True)
    print("Mean Test Accuracy:", round(cv_results['test_score'].mean(), 3))
    print("Mean Train Accuracy:", round(cv_results['train_score'].mean(), 3))
    print("Mean Fit Time:", round(cv_results['fit_time'].mean(), 3))
    print("Mean Score Time:", round(cv_results['score_time'].mean(), 3))

# -- Split the data into train and test sets
X_train, X_test, y_train, y_test = split_data(df)

with mlflow.start_run() as run:
    # -- Get the model
    model = get_model(X_train)
    
    # -- Train the model
    trained_model = train_model(model, X_train, y_train)
    train_score = round(trained_model.score(X_train, y_train), 3)

    mlflow.log_metric('train_score', train_score)
    print("train_score:", train_score)

    evaluation_results = evaluate_model(trained_model, X_test, y_test)
    log_model_and_metrics(trained_model, X_train, y_train, evaluation_results)

    test_score = round(trained_model.score(X_test, y_test), 2)
    print("test_score:", test_score)

    print("accuracy:", round(evaluation_results['accuracy'], 3))
    print("roc_auc:", round(evaluation_results['roc_auc'], 3))
    print("classification_report:", evaluation_results['classification_report'])

    save_and_log_plots(trained_model, X_test, y_test)
    cross_validate_model(trained_model, X_train, y_train)
