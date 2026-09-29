from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def evaluate_classification_model(model, X_test, y_test, model_name):
    print(f"\n[INFO] Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"[OK] {model_name} Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_lstm_model(model, X_test, y_test):
    print("\n[INFO] Evaluating LSTM...")
    X_test_lstm = np.array(X_test)
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))
    y_pred_prob = model.predict(X_test_lstm)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("[OK] LSTM Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return {
        "model": "LSTM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }