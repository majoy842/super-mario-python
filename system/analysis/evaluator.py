from __future__ import annotations


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        accuracy = accuracy_score(y_true, y_pred)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred)

        negative_recall = 0.0
        if "negative" in report:
            negative_recall = float(report["negative"].get("recall", 0.0))

        return {
            "accuracy": float(accuracy),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "negative_recall": negative_recall,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
