import os
import pandas as pd
import importlib
import sys
sys.path.append(os.path.abspath("../"))  # Add directory
import eval_metrics  # Import module

# Reload if modified
importlib.reload(eval_metrics)
from eval_metrics import evaluate_model, custom_threshold_condition

# 🔹 Set the directory containing CSV files
CSV_DIRECTORY = "./predictions/"  # Change this to the actual path

# 🔹 Define thresholds to test
THRESHOLDS = [0.5, 0.1, 0.01]

def find_columns(df):
    """
    Dynamically identifies the correct label, threat, and identity_attack columns.

    Args:
        df (pd.DataFrame): The dataset containing predictions.

    Returns:
        tuple: (label_column, threat_column, identity_attack_column or None)
    """
    label_candidates = [col for col in df.columns if "label" in col.lower()]
    threat_col = None
    identity_attack_col = None

    # Prioritize exact match for 'true_label' or a single 'label' column
    if 'true_label' in df.columns:
        label_col = 'true_label'
    elif 'label' in df.columns:
        label_col = 'label'
    elif len(label_candidates) == 1:
        label_col = label_candidates[0]  # If only one label-related column exists, use it
    else:
        print(f"⚠️ Multiple 'label' columns found: {label_candidates}")
        label_col = input("Enter the correct ground truth label column name: ").strip()  # Ask user for input

    # Find threat & identity_attack columns
    for col in df.columns:
        if "threat" in col.lower():
            threat_col = col
        if "identity_attack" in col.lower():
            identity_attack_col = col

    # Ensure at least 'threat' column exists
    if not label_col or not threat_col:
        raise ValueError(f"Missing required columns in DataFrame: {df.columns}")

    return label_col, threat_col, identity_attack_col  # identity_attack_col may be None


def run_evaluation(df, df_name):
    """
    Runs evaluation on a single dataset.

    Args:
        df (pd.DataFrame): The dataset containing predictions.
        df_name (str): Name of the dataset (for logging).
    """
    try:
        # Identify relevant columns dynamically
        label_col, threat_col, identity_attack_col = find_columns(df)

        # Extract ground truth labels and probabilities
        y_true = df[label_col].to_numpy()

        # Case 1: If both `threat` and `identity_attack` exist
        if identity_attack_col:
            y_probs = df[[threat_col, identity_attack_col]].to_numpy()
            custom_condition = custom_threshold_condition  # Use custom condition for combined labels
        else:
            y_probs = df[[threat_col]].to_numpy()  # Use only threat
            custom_condition = None  # No need for custom condition

        # Run evaluation at different thresholds
        for threshold in THRESHOLDS:
            print(f"\n🚀 Running evaluation for {df_name} at threshold={threshold}...")
            evaluate_model(y_true, y_probs=y_probs, class_labels=["Not Threat", "Threat"], 
                           threshold=threshold, custom_condition=custom_condition)

    except Exception as e:
        print(f"⚠️ Error processing {df_name}: {e}")


def main():
    """
    Main function that loops through CSV files in the directory and evaluates each.
    """
    # 🔹 List all CSV files in the directory
    csv_files = [f for f in os.listdir(CSV_DIRECTORY) if f.endswith(".csv")]

    # 🔹 Loop through CSV files & evaluate each one
    for file in csv_files:
        file_path = os.path.join(CSV_DIRECTORY, file)
        df_name = file.replace(".csv", "")  # Extract file name for reference
        print(f"\n🔹 Processing {df_name}...")

        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Run evaluation for this dataset
            run_evaluation(df, df_name)

        except Exception as e:
            print(f"⚠️ Error loading {df_name}: {e}")


# Run the script when executed directly
if __name__ == "__main__":
    main()
