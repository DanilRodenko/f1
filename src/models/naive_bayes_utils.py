import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate_naive_bayes_train_roc(model, X_train, y_train, plot_title="Naive Bayes - Train ROC"):
    y_probs = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_probs)
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

    print(f"✅ Train ROC AUC (Naive Bayes): {roc_auc:.4f}")
    return roc_auc


def evaluate_naive_bayes_train_confusion(model, X_train, y_train, threshold=0.5):
    y_probs = model.predict_proba(X_train)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    accuracy = accuracy_score(y_train, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n✅ Confusion Matrix (Train, threshold={threshold:.2f}):")
    print(f"   TP: {tp} | FP: {fp}")
    print(f"   FN: {fn} | TN: {tn}")
    print(f"   Accuracy:    {accuracy:.3f}")
    print(f"   Sensitivity: {sensitivity:.3f}")
    print(f"   Specificity: {specificity:.3f}")

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }



def evaluate_naive_bayes_test_roc(model, X_test, y_test, plot_title="Naive Bayes ROC (Test)"):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(plot_title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print(f"✅ Test ROC AUC: {roc_auc:.3f}")
    return roc_auc

