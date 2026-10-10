import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score


def calculate_metrics(labels, probabilities, threshold=.5):
    y = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)), 'precision': float(p), 'sensitivity': float(r), 'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'f1': float(f1), 'roc_auc': float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
