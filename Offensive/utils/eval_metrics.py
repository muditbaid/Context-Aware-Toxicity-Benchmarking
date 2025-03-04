import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, classification_report
)

def evaluate_model(y_true, y_probs=None, y_pred=None, class_labels=None, threshold=0.5, custom_condition=None):
    """
    Evaluates a text classification model with multiple benchmark metrics.
    
    Args:
        y_true (array-like): True labels (0/1 for binary classification).
        y_probs (array-like, optional): Predicted probabilities (for binary classification).
        y_pred (array-like, optional): Direct binary predictions (0/1).
        class_labels (list, optional): List of column names for classification labels.
        threshold (float): Threshold for binary classification.
        custom_condition (function, optional): A function that applies custom thresholding rules.

    Returns:
        None (Displays evaluation metrics and plots).
    """

    # Ensure either y_probs or y_pred is provided
    if y_probs is None and y_pred is None:
        raise ValueError("You must provide either `y_probs` (probabilities) or `y_pred` (binary predictions).")

    # If y_pred is not provided, apply thresholding or custom condition
    if y_pred is None:
        if custom_condition:
            y_pred = custom_condition(y_probs, threshold)  # Apply custom rule
        elif y_probs.ndim == 1 or y_probs.shape[1] == 1:  # Binary classification
            y_pred = (y_probs >= threshold).astype(int)
        else:  # Multi-class classification
            y_pred = np.argmax(y_probs, axis=1)

    # Convert multi-column `y_probs` to a single probability score
    if y_probs is not None and y_probs.ndim == 2:
        y_probs = np.max(y_probs, axis=1)  # Use max probability for final threat label

    # If binary classification, ensure only 2 class labels
    if y_pred.ndim == 1 or len(set(y_pred)) <= 2:
        class_labels = class_labels

    # Compute accuracy, precision, recall, F1-score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    # Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve & AUC Score (Only if `y_probs` is provided for binary classification)
    if y_probs is not None:
        auc = roc_auc_score(y_true, y_probs)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        print(f"✅ AUC Score: {auc:.4f}")

    # Histogram of Predicted Probabilities (Only if `y_probs` is provided)
    if y_probs is not None:
        plt.figure(figsize=(6,5))

        # Plot histogram of final probabilities
        plt.hist(y_probs, bins=30, alpha=0.7, color="blue", edgecolor="black", label=f"{class_labels[1]} Probabilities")

        # Add threshold line
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold = {threshold}")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {class_labels[1]} Probabilities")

        # Add legend with dynamic labels
        plt.legend(loc="upper right")

        plt.show()
        

def custom_threshold_condition(y_probs, threshold, label_indices=[0, 1]):
    """
    Custom rule: If any specified labels exceed the threshold, classify as 1.

    Args:
        y_probs (np.ndarray): Predicted probabilities.
        threshold (float): Decision threshold.
        label_indices (list): List of indices corresponding to labels in `y_probs` that should be considered.

    Returns:
        np.ndarray: Binary classification results based on threshold.
    """
    return (y_probs[:, label_indices] >= threshold).any(axis=1).astype(int)
