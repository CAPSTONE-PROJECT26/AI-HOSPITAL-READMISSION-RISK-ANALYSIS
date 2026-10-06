from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    extension = file_path.suffix.lower()
    if extension == ".csv":
        df = pd.read_csv(file_path)
    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistent downstream processing."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def get_dataset_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a basic data-quality report."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "data_type": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "missing_percent": (df.isna().mean().values * 100).round(2),
            "unique_values": df.nunique(dropna=True).values,
        }
    )
