import pandas as pd

def balance_dataset(df, label_column='true_label'):
    """
    Balances the dataset by downsampling the majority class ('0') to match the count of the minority class ('1').

    Args:
        df (pd.DataFrame): The input dataset containing the label column.
        label_column (str): The name of the column containing binary labels (default is 'true_label').

    Returns:
        pd.DataFrame: A balanced dataset with equal numbers of '0' and '1' labels.
    """
    # Count the occurrences of each label
    label_counts = df[label_column].value_counts()

    # Determine the minority class count (count of '1' labels)
    min_count = label_counts.get(1, 0)  # Get count of '1', default to 0 if missing
    num_zeros = label_counts.get(0, 0)  # Get count of '0', default to 0 if missing

    # If there are more '0' labels, downsample them to match '1' count
    if num_zeros > min_count and min_count > 0:
        df_ones = df[df[label_column] == 1]
        df_zeros = df[df[label_column] == 0].sample(n=min_count, random_state=42)
        
        # Combine balanced dataset
        balanced_df = pd.concat([df_ones, df_zeros]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        balanced_df = df.copy()  # No downsampling needed if already balanced or missing '1' labels

    return balanced_df
