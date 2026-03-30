import pandas as pd

def load_dataset(file_path: str):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_dataset(data: pd.DataFrame):
    #Cleans the dataset before split
    df = data.copy()
    df.columns = df.columns.str.strip()
    dropColumns = ['category', 'specific_class', 'ID']
    dropExisting = [col for col in dropColumns if col in df.columns]
    if (dropExisting):
        df = df.drop(columns = dropExisting)
    
    #remove duplicate
    df = df.drop_duplicates()

    constantColumns = [col for col in df.columns if col != 'label' and df[col].nunique(dropna=False) <= 1]
    if constantColumns:
        df = df.drop(columns = constantColumns)
        print(f"Removed constant col: {constantColumns}")
        
    df = df.dropna()
    return df
    
def process_benign_dataset(benign_data):
    benign_data = clean_dataset(benign_data)
    X = benign_data.drop(columns=['label'])
    X = X.select_dtypes(include=['number'])
    y = benign_data['label']
    
    # convert benign labels to 0 and DoS labels to 1
    y = y.astype(str).str.strip().apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
        
    return X, y

def process_dos_dataset(dos_data):
    dos_data = clean_dataset(dos_data)
    X = dos_data.drop(columns=['label'])
    X = X.select_dtypes(include=['number'])
    y = dos_data['label']
    
    # convert DoS labels to 1 and benign labels to 0
    y = y.astype(str).str.strip().apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
    
    return X, y

def combine_datasets(benign_X, benign_y, dos_X, dos_y):
    """
    Combine benign and DoS datasets into a single dataset.
    
    Args:
        benign_X: Features from benign dataset
        benign_y: Labels from benign dataset
        dos_X: Features from DoS dataset
        dos_y: Labels from DoS dataset
    
    Returns:
        Tuple of combined features (X) and labels (y)
    """
    X = pd.concat([benign_X, dos_X], ignore_index=True)
    y = pd.concat([benign_y, dos_y], ignore_index=True)
    return X, y

def random_combine_datasets(benign_X, benign_y, dos_X, dos_y):
    """
    Combine benign and DoS datasets into a single dataset with random shuffling.
    
    Args:
        benign_X: Features from benign dataset
        benign_y: Labels from benign dataset
        dos_X: Features from DoS dataset
        dos_y: Labels from DoS dataset
    
    Returns:
        Tuple of combined features (X) and labels (y)
    """
    combined_X = pd.concat([benign_X, dos_X], ignore_index=True)
    combined_y = pd.concat([benign_y, dos_y], ignore_index=True)
    
    combined = pd.concat([combined_X, combined_y.rename("label_binary")], axis = 1)
    # Shuffle the combined dataset
    combined = combined.sample(frac = 1, random_state=42).reset_index(drop = True)
    X = combined.drop(columns = ['label_binary'])
    y = combined['label_binary']
    
    return X, y

def remove_cross_duplicates(X, y):
    #removes identical rows that appear with different labels
    combined = X.copy()
    combined['label_binary'] = y.values

    duplicated_features = combined.drop(columns=['label_binary']).duplicated(keep=False)
    possible_conflicts = combined[duplicated_features]

    if not possible_conflicts.empty:
        grouped = possible_conflicts.groupby(list(X.columns))['label_binary'].nunique()
        conflicting_index = grouped[grouped > 1].index

        if len(conflicting_index) > 0:
            feature_only = combined.drop(columns=['label_binary'])
            mask = feature_only.apply(tuple, axis=1).isin(conflicting_index)
            removed_count = mask.sum()
            combined = combined.loc[~mask].reset_index(drop=True)
            print(f"Removed {removed_count} conflicting duplicate rows across classes.")

    X_clean = combined.drop(columns=['label_binary'])
    y_clean = combined['label_binary']
    return X_clean, y_clean

def get_dataset() -> None | tuple[pd.DataFrame, pd.Series]:
    """
    Load and process the benign and DoS datasets, 
    then combine them into a single dataset.
    
    Return:
        the combined features and labels as a tuple (X, y) if successful, or None if there was an error.
    """
    benign_data = load_dataset('dataset/decimal_benign.csv')
    dos_data = load_dataset('dataset/decimal_DoS.csv')
    
    if benign_data is None:
        print("Failed to load benign dataset.")
        return None
    
    if dos_data is None:
        print("Failed to load DoS dataset.")
        return None
    
    benign_X, benign_y = process_benign_dataset(benign_data)
    dos_X, dos_y = process_dos_dataset(dos_data)
    
    X, y = random_combine_datasets(benign_X, benign_y, dos_X, dos_y)
    X, y = remove_cross_duplicates(X, y)
    print("Final columns:")
    print(X.columns.tolist())
    print(f"Final dataset shape: X = {X.shape}, y={y.shape}")
    print("Distribution:")
    print(y.value_counts())
    return X, y

# For testing the dataset processing
if __name__ == "__main__":
    dataset = get_dataset()
    if dataset is not None:
        X, y = dataset
        print("Dataset processed successfully.")
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        
        print("Sample features:")
        print(X.head())
        print("Sample labels:")
        print(y.head())
    else:
        print("Failed to process dataset.")