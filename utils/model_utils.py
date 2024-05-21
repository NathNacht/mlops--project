import mlflow
from sklearn.metrics import roc_auc_score, classification_report

def train_model(model_pipe, X_train, y_train):
    model_pipe.fit(X_train, y_train)
    return model_pipe

def evaluate_model(trained_model, X_test, y_test):
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
