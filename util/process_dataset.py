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
    #df = df.drop_duplicates()

    constantColumns = [col for col in df.columns if col != 'label' and df[col].nunique(dropna=False) <= 1]
    if constantColumns:
        df = df.drop(columns = constantColumns)
        print(f"Removed constant col: {constantColumns}")
        
    df = df.dropna()
    return df
    
def process_dataset(data: pd.DataFrame):
    df = clean_dataset(data)
    X = data.drop(columns=['label'])
    X = X.select_dtypes(include=['number'])
    
    # convert benign labels to 0 and DoS labels to 1
    y = df['label'].astype(str).str.strip().apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
        
    return X, y

def combine_datasets(feature_sets, label_sets, shuffle = True, random_state = 42):
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
    X = pd.concat(feature_sets, ignore_index=True)
    y = pd.concat(label_sets, ignore_index=True)

    if 'ID' in X.columns:
        print("ID in dataset, removing")
        X = X.drop(columns=['ID'])
    if shuffle:
        combined = pd.concat([X, y.rename("label_binary")], axis = 1)
        combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
        X = combined.drop(columns=['label_binary'])
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
    benign= 'dataset/decimal_benign.csv'
    
    attack_files = [
        'dataset/decimal_DoS.csv',
        'dataset/decimal_spoofing-GAS.csv',
        'dataset/decimal_spoofing-RPM.csv',
        'dataset/decimal_spoofing-SPEED.csv',
        'dataset/decimal_spoofing-STEERING_WHEEL.csv',
    ]
    benign_data = load_dataset(benign)
    if benign_data is None:
        print("Failed to load benign dataset")
        return None
    
    feature_sets = []
    label_sets = []

    benign_X, benign_y = process_dataset(benign_data)
    feature_sets.append(benign_X)
    label_sets.append(benign_y)

    for file_path in attack_files:
        attack_data = load_dataset(file_path)
        if attack_data is None:
            print(f"skip dataset: {file_path}")
            continue
        attack_X, attack_y = process_dataset(attack_data)
        feature_sets.append(attack_X)
        label_sets.append(attack_y)

    if len(feature_sets) < 2:
        print("No attack datasets loaded")
        return None
    
    X, y = combine_datasets(feature_sets, label_sets, shuffle=True, random_state = 42)
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