from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def plot_roc_curve(model, X, y, title="ROC Curve"):
    probs = model.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y, probs)
    auc_score = roc_auc_score(y, probs)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot()
    plt.title(title)
    plt.grid()
    plt.show()

    print(f"AUC: {auc_score:.4f}")


def evaluate_model_classification(model, X, y, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y, preds)
    print(f"✅ Accuracy: {acc*100:.2f}%\n")

    print("=== Classification Report ===")
    print(classification_report(y, preds))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y, preds))


def find_best_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr  # Youden's J statistic (J = TPR - FPR)
    best_index = j_scores.argmax()
    best_threshold = thresholds[best_index]

    print(f"✅ Best threshold: {best_threshold:.4f}")
    print(f"   Sensitivity (TPR): {tpr[best_index]:.3f}")
    print(f"   Specificity (TNR): {1 - fpr[best_index]:.3f}")

    return best_threshold
