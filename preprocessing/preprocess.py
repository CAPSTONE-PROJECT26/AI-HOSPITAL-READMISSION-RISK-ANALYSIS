from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def encode_target(y: pd.Series, mapping: dict | None = None) -> pd.Series:
    """Convert a binary target into 0/1 values."""
    y = y.copy()

    if y.dtype == bool:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        unique_values = set(y.dropna().unique())
        if unique_values.issubset({0, 1}):
            return y.astype(int)

    if mapping is not None:
        encoded = y.map(mapping)
        if encoded.isna().any():
            raise ValueError("Some target values were not present in TARGET_MAPPING.")
        return encoded.astype(int)

    raise ValueError(
        "Target is not already binary 0/1. Provide TARGET_MAPPING in "
        "preprocessing/run_preprocessing.py."
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
):
    """Split data into 70% train, 15% validation, and 15% test."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_ratio = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def save_processed_data(X, y, output_directory: str | Path, split_name: str, feature_names):
    """Save processed features and target to CSV files."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X, columns=feature_names).to_csv(
        output_directory / f"X_{split_name}.csv", index=False
    )
    pd.DataFrame({"readmission_30d": y}).to_csv(
        output_directory / f"y_{split_name}.csv", index=False
    )


def run_pipeline(
    input_path: str,
    target_column: str,
    target_mapping: dict | None = None,
    drop_columns: list[str] | None = None,
):
    """Run the complete preprocessing pipeline."""
    from preprocessing.load_data import clean_column_names, load_dataset

    df = load_dataset(input_path)
    print(f"Original dataset shape: {df.shape}")

    df = clean_column_names(df)
    target_column = target_column.strip().lower()

    if drop_columns:
        df = df.drop(columns=[c.lower() for c in drop_columns], errors="ignore")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\nAvailable columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = encode_target(df[target_column], mapping=target_mapping)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    output_dir = Path("data/processed")
    save_processed_data(X_train_processed, y_train, output_dir, "train", feature_names)
    save_processed_data(X_val_processed, y_val, output_dir, "validation", feature_names)
    save_processed_data(X_test_processed, y_test, output_dir, "test", feature_names)

    report = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rows": [len(X_train), len(X_val), len(X_test)],
            "positive_rate": [
                y_train.mean(),
                y_val.mean(),
                y_test.mean(),
            ],
        }
    )
    report.to_csv(output_dir / "split_report.csv", index=False)

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    print("\nPreprocessing completed successfully.")
    print(f"Train:      {X_train_processed.shape}")
    print(f"Validation: {X_val_processed.shape}")
    print(f"Test:       {X_test_processed.shape}")
    print("Saved to data/processed/")
