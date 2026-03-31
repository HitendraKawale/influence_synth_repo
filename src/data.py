from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

import pandas as pd


@dataclass
class PreparedTabularData:
    df: pd.Dataframe
    target_mapping: dict[Any, Any]
    categorical_cols: list[str]
    numerical_cols: list[str]


def _encode_binary_target(
    series: pd.Series,
    positive_label: Any | None = None,
) -> tuple[pd.Series, dict[Any, int]]:
    clean = series.dropna()
    unique_values = list(pd.unique(clean))

    if len(unique_values) != 2:
        raise ValueError(
            f"Expected a binary target, but found {len(unique_values)} unique values: {unique_values}"
        )

    if positive_label is not None:
        if positive_label not in unique_values:
            raise ValueError(
                f"positive_label={positive_label!r} not found in target values {unique_values}"
            )
        negative_label = [v for v in unique_values if v != positive_label][0]
        mapping = {negative_label: 0, positive_label: 1}
    else:
        sorted_values = sorted(unique_values, key=lambda x: str(x))
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}

    encoded = series.map(mapping)
    return encoded.astype("Int64"), mapping


def load_and_prepare_binary_tabular_data(
    data_path: str | Path,
    target_col: str,
    drop_cols: list[str] | None = None,
    positive_label: Any | None = None,
) -> PreparedTabularData:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    drop_cols = drop_cols or []
    missing_drop_cols = [col for col in drop_cols if col not in df.columns]
    if missing_drop_cols:
        raise ValueError(f"Columns not found in dataset: {missing_drop_cols}")

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in dataset.")

    df = df.drop(columns=drop_cols).copy()
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    encoded_target, mapping = _encode_binary_target(
        df[target_col], positive_label=positive_label
    )
    df[target_col] = encoded_target.astype(int)

    feature_df = df.drop(columns=[target_col])

    categorical_cols = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_cols = [col for col in feature_df.columns if col not in categorical_cols]

    return PreparedTabularData(
        df=df,
        target_mapping=mapping,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )


def coerce_dataframe_to_reference_schema(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Best-effort schema alignment for synthetic samples."""
    aligned = df.copy()

    missing_cols = [col for col in reference_df.columns if col not in aligned.columns]
    if missing_cols:
        raise ValueError(f"Synthetic dataframe is missing columns: {missing_cols}")

    aligned = aligned[reference_df.columns].copy()

    for col in reference_df.columns:
        ref_dtype = reference_df[col].dtype

        if pd.api.types.is_numeric_dtype(ref_dtype):
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            if pd.api.types.is_integer_dtype(ref_dtype):
                aligned[col] = aligned[col].round()

        elif pd.api.types.is_bool_dtype(ref_dtype):
            aligned[col] = aligned[col].astype("boolean")

        else:
            aligned[col] = aligned[col].astype("object")

    # Target cleanup
    aligned[target_col] = pd.to_numeric(aligned[target_col], errors="coerce").round()
    valid_target_values = set(reference_df[target_col].unique().tolist())
    aligned = aligned[aligned[target_col].isin(valid_target_values)].copy()

    if aligned.empty:
        raise ValueError("No synthetic rows remained after target/schema alignment.")

    aligned[target_col] = aligned[target_col].astype(reference_df[target_col].dtype)
    aligned = aligned.reset_index(drop=True)

    return aligned
