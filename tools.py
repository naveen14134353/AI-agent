import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# LOAD DATASET
def load_dataset(file_path):
    return pd.read_csv(file_path)


# DATASET INFORMATION
def dataset_info(df):

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    lines = []

    lines.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    lines.append(
        f"Numeric columns ({len(numeric_columns)}): "
        + ", ".join(numeric_columns)
    )

    lines.append(
        f"Categorical columns ({len(categorical_columns)}): "
        + ", ".join(categorical_columns)
    )

    lines.append("\nColumn details:")

    for column in df.columns:

        dtype = str(df[column].dtype)
        missing = int(df[column].isna().sum())

        lines.append(
            f"- {column}: dtype={dtype}, missing={missing}"
        )

    return "\n".join(lines)


# STATISTICAL SUMMARY
def statistical_summary(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return "No numerical columns found."

    summary = numeric_df.describe().T

    lines = ["Descriptive statistics for numerical columns:"]

    for column, row in summary.iterrows():

        lines.append(
            f"- {column}: "
            f"mean={row['mean']:.2f}, "
            f"std={row['std']:.2f}, "
            f"min={row['min']:.2f}, "
            f"max={row['max']:.2f}"
        )

    return "\n".join(lines)


# CORRELATION ANALYSIS
def correlation_analysis(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return "Not enough numerical columns for correlation analysis."

    corr = numeric_df.corr()

    # Extract unique column pairs
    pairs = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    ).stack()

    if pairs.empty:
        return "No valid correlations found."

    # Sort by absolute correlation strength
    pairs = pairs.reindex(
        pairs.abs().sort_values(ascending=False).index
    )

    lines = ["Strongest correlations:"]

    for (col1, col2), value in pairs.head(10).items():

        lines.append(
            f"- {col1} ↔ {col2}: correlation={value:.3f}"
        )

    return "\n".join(lines)


# HISTOGRAM
def create_histogram(df, column):

    if column not in df.columns:
        return f"Column '{column}' does not exist."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column].dropna(),
        bins=20
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")

    plt.tight_layout()

    path = f"{column}_histogram.png"

    plt.savefig(path)
    plt.close()

    return f"Histogram created successfully: {path}"