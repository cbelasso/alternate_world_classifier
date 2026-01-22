"""
Data I/O Utility Functions

Provides functions for reading and writing various data formats
including Excel, CSV, JSON, and more.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_tabular_file(
    filepath: str | Path, sheet_name: str | int | None = None
) -> pd.DataFrame:
    """
    Read tabular data from Excel, CSV, or JSON file.

    Args:
        filepath: Path to the input file
        sheet_name: For Excel files, the sheet name or index (0-based).
                    If None, reads the first sheet. Ignored for CSV/JSON.

    Returns:
        pandas DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, sheet_name=sheet_name)
    elif suffix == ".csv":
        return pd.read_csv(filepath)
    elif suffix == ".json":
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx, .xls, .csv, or .json")


def save_json(
    data: dict[str, Any] | list[Any],
    filepath: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save data to a JSON file with pretty formatting.

    Args:
        data: Dictionary or list to save
        filepath: Output path for JSON file
        indent: Number of spaces for indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)

    Returns:
        Path to saved file
    """
    filepath = Path(filepath)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    return filepath


def load_json(filepath: str | Path) -> dict[str, Any] | list[Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON data (dict or list)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_excel_sheet_names(filepath: str | Path) -> list[str]:
    """
    Get list of sheet names from an Excel file.

    Args:
        filepath: Path to Excel file

    Returns:
        List of sheet names
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    xlsx = pd.ExcelFile(filepath)
    return xlsx.sheet_names


def save_dataframe(
    df: pd.DataFrame, filepath: str | Path, index: bool = False, **kwargs
) -> Path:
    """
    Save DataFrame to file based on extension.

    Args:
        df: pandas DataFrame to save
        filepath: Output path (extension determines format)
        index: Whether to include row index (default: False)
        **kwargs: Additional arguments passed to pandas save method

    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df.to_excel(filepath, index=index, **kwargs)
    elif suffix == ".csv":
        df.to_csv(filepath, index=index, **kwargs)
    elif suffix == ".json":
        df.to_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx, .xls, .csv, or .json")

    return filepath
