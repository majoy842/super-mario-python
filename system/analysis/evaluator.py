from __future__ import annotations


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
