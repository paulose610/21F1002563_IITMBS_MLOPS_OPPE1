import pandas as pd
from pathlib import Path

def preprocess_all(input_dir: str = "raw_data", output_dir: str = "data", n_rows: int = 11_000):
    """
    Preprocesses the first n_rows (default=11,000) of each CSV in input_dir
    and saves the processed result as a CSV (same name) in output_dir.

    Features:
      - rolling_avg_10 : 10-min moving average of 'close'
      - volume_sum_10  : 10-min total traded volume
      - target         : 1 if close(t+5) > close(t)
      - stock_name     : derived from filename
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in input directory.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name} ...")
        try:
            # --- Load only first n_rows ---
            df = pd.read_csv(csv_file, nrows=n_rows)
            if df.empty:
                print(f"⚠️  Skipped {csv_file.name} (empty file).")
                continue

            # --- Clean ---
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.ffill(inplace=True)

            # --- Add entity ---
            stock_name = Path(csv_file).stem.split("_")[0]
            df["stock_name"] = stock_name

            # --- Feature engineering ---
            df.set_index("timestamp", inplace=True)
            df["rolling_avg_10"] = df["close"].rolling("10min", min_periods=1).mean()
            df["volume_sum_10"] = df["volume"].rolling("10min", min_periods=1).sum()

            # Target: 1 if close(t+5) > close(t)
            df["close_5min_future"] = df["close"].shift(-5)
            df["target"] = (df["close_5min_future"] > df["close"]).astype(int)

            # --- Cleanup ---
            df.drop(columns=["close_5min_future"], inplace=True)
            df.dropna(subset=["rolling_avg_10", "volume_sum_10", "target"], inplace=True)
            df.reset_index(inplace=True)

            # --- Save as CSV ---
            output_path = output_dir / csv_file.name
            df.to_csv(output_path, index=False)
            print(f"✅ Saved processed file: {output_path} ({df.shape})")

        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")

if __name__ == "__main__":
    preprocess_all()

