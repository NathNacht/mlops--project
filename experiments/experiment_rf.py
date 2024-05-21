import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate, train_test_split

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


# Set Auto logging for Scikit-learn flavor 
# mlflow.sklearn.autolog()

with mlflow.start_run() as run:

    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    # -- Split the data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -- Get the model
    model = get_model(X_train)

    # -- Train the model
    trained_model = train_model(model, X_train, y_train)

    # -- Evaluate the model
    evaluation_results = evaluate_model(trained_model, X_test, y_test)

    mlflow.log_metric('accuracy', evaluation_results['accuracy'])
    mlflow.log_metric('roc_auc', evaluation_results['roc_auc'])

    report = evaluation_results['classification_report']
    with open("classification_report.txt", "w") as f:
        f.write(report)
    
    mlflow.log_artifact("classification_report.txt", artifact_path="metrics")

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
    plt.savefig("metrics/precision_recall_curve.png")
    plt.close()
    mlflow.log_artifact("metrics/precision_recall_curve.png", artifact_path="metric_graphs")


    # -- Log the model
    mlflow.sklearn.log_model(
        sk_model = model, 
        artifact_path="random_forest_classifier",
        conda_env=None)



    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]
    
    # # Calculate metrics
    # accuracy = (y_pred == y_test).mean()
    # roc_auc = roc_auc_score(y_test, y_prob)
    # report = classification_report(y_test, y_pred)
    # # y_pred = evaluation_results['y_pred']

    # # log the precision-recall curve
    # fig_pr = plt.figure()
    # pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    # plt.title('Precision-Recall Curve')
    # plt.legend()

    # mlflow.log_figure(fig_pr, 'precision_recall_curve.png')



    # Log the experiment
    # params = {'model_type': 'RandomForest', 'n_estimators': 100, 'max_depth': 5}
    # metrics = {'accuracy': accuracy}
    # log_experiment(params, metrics)
