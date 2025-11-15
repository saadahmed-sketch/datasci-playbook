from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Accuracy
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Confusion Matrix
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Classification Report
def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

# ROC-AUC
def get_roc_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)

# PR-AUC
def get_pr_auc(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

# Optional: Plot ROC Curve
def plot_roc_curve(y_true, y_proba, model_name="Model"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {get_roc_auc(y_true, y_proba):.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Optional: Plot PR Curve
def plot_pr_curve(y_true, y_proba, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, label=f"{model_name} (AUC = {get_pr_auc(y_true, y_proba):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
