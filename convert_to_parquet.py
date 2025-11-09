import pandas as pd
from pathlib import Path

def csv_to_parquet(csv_path="data/data.csv", parquet_path="source/data.parquet"):
    """
    Converts a CSV file to Parquet format.
    Input:  data/data.csv
    Output: source/data.parquet
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Saving Parquet to: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    print("✅ Conversion complete.")

if __name__ == "__main__":
    csv_to_parquet()

