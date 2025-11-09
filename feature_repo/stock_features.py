from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64, String
from datetime import timedelta

# Entity
stock = Entity(name="stock_name", join_keys=["stock_name"], value_type=String)

# Source (train data)
stock_source = FileSource(
    path="../source/train.parquet",
    timestamp_field="timestamp",
)

# Feature View
stock_fv = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
)
