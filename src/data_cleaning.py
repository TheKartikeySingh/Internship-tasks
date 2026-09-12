import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df
