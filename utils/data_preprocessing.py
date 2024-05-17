import pandas as pd
import os

# added to test script individually
filename = 'BankChurners.csv'

def load_data(filename):
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', filename)
    return pd.read_csv(data_path)

def preprocess_data(df):
    df = df[df.columns[:-2]]
    df = df.drop(['CLIENTNUM'], axis=1)
    return df

def save_cleaned_data(df):
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'cleaned_data.csv')
    df.to_csv(data_path, index=False)

df = load_data(filename)
df = preprocess_data(df)
save_cleaned_data(df)