import pandas as pd
from pathlib import Path

def fix_timestamps(parquet_path: Path):
    if not parquet_path.exists():
        print(f"⚠️ Skipping — {parquet_path} not found.")
        return

    print(f"\nReading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    # Convert to proper timezone-naive datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    print("✅ dtype after fix:", df["timestamp"].dtype)
    print(df["timestamp"].head(3))

    # Overwrite same file
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Fixed and saved back to {parquet_path}")

def main():
    base_dir = Path("source")
    for file_name in ["train.parquet", "validate.parquet"]:
        fix_timestamps(base_dir / file_name)

if __name__ == "__main__":
    main()
