import subprocess
import pandas as pd
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run shell commands and print output."""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr.strip():
        print("Warning:", result.stderr)
    result.check_returncode()

def main():
    base_dir = Path(__file__).resolve().parent
    feature_repo_dir = base_dir / "feature_repo"
    train_parquet = base_dir / "source" / "train.parquet"
    
    # 1. Run preprocessing
    run_cmd("python preprocess.py")

    # 2. Run augment.sh (merges all CSVs in data/)
    run_cmd("bash augment.sh")

    # 3. Convert merged CSV to Parquet
    run_cmd("python convert_to_parquet.py")

    # 4. fixing timestamp of parquet
    run_cmd("python fix_parquet.py")
    
    # 5. Run feast apply inside feature_repo
    run_cmd("feast apply", cwd=feature_repo_dir)
    
    
    
    # 6. Detect min and max timestamps from source/train.parquet
    print("\nDetecting timestamp range from source/train.parquet ...")
    df = pd.read_parquet(train_parquet, columns=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()

    print(f"Min timestamp: {min_ts}")
    print(f"Max timestamp: {max_ts}")

    # 7. Run feast materialize with detected timestamps
    cmd = f'feast materialize "{min_ts.isoformat()}" "{max_ts.isoformat()}"'
    run_cmd(cmd, cwd=feature_repo_dir)

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()


