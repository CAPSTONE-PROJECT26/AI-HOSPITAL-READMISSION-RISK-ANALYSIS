from preprocessing.preprocess import run_pipeline

# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------
# Put the actual dataset in data/raw/ and update this path.
INPUT_FILE = "data/raw/patients.csv"

# Change this after inspecting the actual dataset.
TARGET_COLUMN = "readmission_30d"

# Use only when the target is not already encoded as 0/1.
# Example:
# TARGET_MAPPING = {"yes": 1, "no": 0}
TARGET_MAPPING = None

# Add columns that should not enter the model after the team agrees.
DROP_COLUMNS: list[str] = []


if __name__ == "__main__":
    run_pipeline(
        input_path=INPUT_FILE,
        target_column=TARGET_COLUMN,
        target_mapping=TARGET_MAPPING,
        drop_columns=DROP_COLUMNS,
    )
