import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import statsmodels.api as sm

def run_logistic_regression_from_data(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model



def check_logistic_significance(X, y):
    X_with_const = sm.add_constant(X)
    model = sm.Logit(y, X_with_const).fit()
    print(model.summary())

def evaluate_roc_auc(model, X_test, y_test, plot_title="ROC Curve"):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(plot_title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print(f"✅ Test ROC AUC: {roc_auc:.3f}")
    return roc_auc



def evaluate_threshold_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"✅ Threshold: {threshold:.4f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Sensitivity (TPR): {sensitivity:.3f}")
    print(f"   Specificity (TNR): {specificity:.3f}")

    return accuracy, sensitivity, specificity
