import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix from true labels and predicted labels.
    """
    # Initialize variables for TP, TN, FP, FN
    TP = TN = FP = FN = 0
    
    # Iterate through the true labels and predictions
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:  # True Positive
            TP += 1
        elif true == 0 and pred == 0:  # True Negative
            TN += 1
        elif true == 0 and pred == 1:  # False Positive
            FP += 1
        elif true == 1 and pred == 0:  # False Negative
            FN += 1
    
    return TP, TN, FP, FN

def accuracy(TP, TN, FP, FN):
    """
    Compute accuracy from confusion matrix components.
    """
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    """
    Compute precision from confusion matrix components.
    """
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall(TP, FN):
    """
    Compute recall from confusion matrix components (same as sensitivity).
    """
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score(precision, recall):
    """
    Compute F1 score from precision and recall.
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Example usage
y_true = [0, 1, 1, 0, 1, 0, 0, 1]  # True labels
y_pred = [0, 0, 1, 0, 1, 1, 0, 1]  # Predicted labels

# Create confusion matrix
TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

# Calculate metrics
acc = accuracy(TP, TN, FP, FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1 = f1_score(prec, rec)
sens = rec  # Sensitivity is the same as recall

# Display results
print("Confusion Matrix:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print("\nMetrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (Sensitivity): {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
