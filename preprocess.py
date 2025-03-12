import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def process_data(path):
    df = pd.read_csv(path)
    # drop id column as it is not necessary
    df.drop('hsi_id', axis=1, inplace=True)
    # Use sklearn Imputer insted of dropping rows with null values but in our dataset there is no null values
    df.dropna(inplace=True)
    target = "vomitoxin_ppb"
    X = df.drop(columns=[target])
    y = df[target]
    # Use standard scaler for preprocessing step could use (MinMaxScaler or Normalization) but standardSCaler is better
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return (X,y),(X_scaled,y)

