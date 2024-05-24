from sklearn.metrics import roc_auc_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from typing import Dict, Any

def train_model(model_pipe: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
    """
    Train the model pipeline.
    
    Parameters:
    model_pipe (ImbPipeline): The pipeline containing preprocessing and model.
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    
    Returns:
    ImbPipeline: The trained model pipeline.
    """
    model_pipe.fit(X_train, y_train)
    return model_pipe

def evaluate_model(trained_model: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Parameters:
    trained_model (ImbPipeline): The trained model pipeline.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test labels.
    
    Returns:
    Dict[str, Any]: A dictionary containing evaluation metrics and predictions.
    """
    y_pred = trained_model.predict(X_test)
    y_prob = trained_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'y_pred': y_pred
    }
