import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow setup
mlflow.set_tracking_uri("http://136.112.91.129:8100")
EXPERIMENT_NAME = "OPPE1 stock exp v2"
client = MlflowClient(mlflow.get_tracking_uri())

# Verify the experiment exists
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
assert exp is not None, f"Experiment '{EXPERIMENT_NAME}' not found!"
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Using experiment: {EXPERIMENT_NAME}")
print(f"Artifact location: {exp.artifact_location}")

MODEL_NAME = "OPPE1-stock-classifier"
best_model_uri = None


def test_get_best_model():
    """
    Finds the best-performing model version (highest accuracy)
    and tags it as 'best_model'.
    """
    global best_model_uri

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert versions, f"No registered versions found for model {MODEL_NAME}"

    best_version = None
    best_accuracy = 0

    for v in versions:
        run = client.get_run(v.run_id)
        acc = run.data.metrics.get("accuracy", 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_version = v

    assert best_version, "No model version with accuracy found."

    best_model_uri = f"models:/{MODEL_NAME}/{best_version.version}"
    print(f"Best model: {best_model_uri} | Accuracy: {best_accuracy}")

    # Tag as best model in MLflow registry
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=best_version.version,
        key="status",
        value="best_model"
    )

    print(f"Tagged model version {best_version.version} as 'best_model'.")


def test_model_performance():
    """
    Loads the best model and evaluates it on first 100 rows of validation data.
    """
    global best_model_uri
    assert best_model_uri, "best_model_uri not set — run test_get_best_model() first!"

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, "source", "validate.parquet")
    metrics_path = os.path.join(script_dir, "metadata", "metrics.csv")

    # Load model
    model = mlflow.sklearn.load_model(best_model_uri)
    print("Model loaded successfully from registry!")

    # Load validation data
    df = pd.read_parquet(data_path).head(100)
    X_val = df.drop(columns=["target", "timestamp", "stock_name"])
    y_val = df["target"]

    # Evaluate
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted"),
        "recall": recall_score(y_val, y_pred, average="weighted"),
        "f1_score": f1_score(y_val, y_pred, average="weighted"),
    }

    # Save metrics
    os.makedirs(os.path.join(script_dir, "metadata"), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Metrics saved successfully:", metrics)


if __name__ == "__main__":
    test_get_best_model()
    test_model_performance()

