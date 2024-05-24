import pandas as pd
import os

# added to test script individually
filename = 'BankChurners.csv'

def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    filename (str): The name of the CSV file to load.
    
    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', filename)
    return pd.read_csv(data_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by selecting certain columns and dropping the CLIENTNUM column.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    df = df[df.columns[:-2]]
    df = df.drop(['CLIENTNUM'], axis=1)
    return df

def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned data to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The cleaned dataframe to save.
    
    Returns:
    None
    """
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'cleaned_data.csv')
    df.to_csv(data_path, index=False)

# df = load_data(filename)
# df = preprocess_data(df)
# save_cleaned_data(df)