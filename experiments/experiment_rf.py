import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate, train_test_split
from mlflow.models.signature import infer_signature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from utils.data_preprocessing import load_data, preprocess_data
from utils.model_utils import train_model, evaluate_model
from models.model_rf import get_model

# setting the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment to "Sklearn Model"
mlflow.set_experiment("Sklearn RandomForestClassifier")

# Load and preprocess data
filename = 'BankChurners.csv'
df = load_data(filename)
df = preprocess_data(df)

with mlflow.start_run() as run:

    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    # -- Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -- Get the model
    model = get_model(X_train)

    # -- Train the model
    trained_model = train_model(model, X_train, y_train)

    # -- Model Signature (prints input and output schema of the model)
    model_signature = infer_signature(X_train, y_train, params={'model_type': 'RandomForest'})
    print(model_signature.to_dict())
    
    # -- Log the model
    mlflow.sklearn.log_model(
        sk_model = trained_model, 
        artifact_path="random_forest_classifier_trained",
        conda_env=None,
        signature=model_signature)

    # -- Evaluate the model
    evaluation_results = evaluate_model(trained_model, X_test, y_test)

    mlflow.log_metric('accuracy', evaluation_results['accuracy'])
    mlflow.log_metric('roc_auc', evaluation_results['roc_auc'])

    report = evaluation_results['classification_report']
    with open("metrics/classification_report.txt", "w") as f:
        f.write(report)
    
    mlflow.log_artifact("metrics/classification_report.txt", artifact_path="metrics")

    print("accuracy: ", evaluation_results['accuracy'])
    print("roc_auc: ", evaluation_results['roc_auc'])
    print("classification_report: ", evaluation_results['classification_report'])
    
    # -- Plot the ROC and PR curves
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

    # DOING A PREDICTION FROM A stored MODEL
    # logged_model = 'runs:/b474af18c1d44ee38e905b9a0c5852ee/random_forest_classifier_trained'
    # # Load model as a PyFuncModel.
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    # # Predict on a Pandas DataFrame.
    # import pandas as pd
    # y_pred = loaded_model.predict(pd.DataFrame(X_test))

    # print(y_pred)

