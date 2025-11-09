import pandas as pd
from pathlib import Path

def csv_to_parquet_split(
    csv_path="data/data.csv",
    output_dir="source",
    n_train=10_000
):
    """
    Converts a CSV file into two Parquet files:
      - train.parquet: first n_train rows
      - validate.parquet: remaining rows

    Input:  data/data.csv
    Output: source/train.parquet, source/validate.parquet
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Split
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()

    # Save as Parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validate.parquet"
    
    df["timestamp"] = pd.to_datetime(df["timestamp"],errors="coerce").dt.tz_localize(None)
    
    print(f"Saving train data → {train_path} ({len(train_df)} rows)")
    train_df.to_parquet(train_path, index=False)

    print(f"Saving validation data → {val_path} ({len(val_df)} rows)")
    val_df.to_parquet(val_path, index=False)

    print("✅ Conversion complete.")

if __name__ == "__main__":
    csv_to_parquet_split()

