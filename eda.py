
import pandas as pd

# Load retail sales dataset
df = pd.read_csv("dataset/train.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDATE RANGE")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print("Start date:", df["date"].min())
    print("End date:", df["date"].max())

print("\nSALES SUMMARY")
if "sales" in df.columns:
    print(df["sales"].describe())

print("\nEDA COMPLETED SUCCESSFULLY!")