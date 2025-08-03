import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    # Read CSV and clean column names
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    # Drop Customer_ID, encode Gender
    df = df.drop(columns=['Customer_ID'])
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df



 


