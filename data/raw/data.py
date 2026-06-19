import pandas as pd

df = pd.read_csv("data/raw/ai4i.csv")

print(df.shape)
print(df.head())
print(df.columns)