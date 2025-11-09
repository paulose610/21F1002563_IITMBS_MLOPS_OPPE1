import os
import pandas as pd

def test_data_shape_and_columns():
    """
    Validate that the validation dataset has expected structure and is non-empty.
    """

    # Locate validation data
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, "source", "validate.parquet")

    # Load data
    df = pd.read_parquet(data_path)

    # Expected columns
    expected_columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "stock_name", "rolling_avg_10", "volume_sum_10", "target"
    ]

    # Column validation
    assert all(col in df.columns for col in expected_columns), \
        f"Missing columns: {set(expected_columns) - set(df.columns)}"

    # Non-empty and reasonable shape
    assert len(df) > 0, "Validation dataset is empty"
    assert df.shape[1] == len(expected_columns), "Unexpected number of columns"

if __name__ == "__main__":
    test_data_shape_and_columns()

