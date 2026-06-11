"""
SMD Data Preprocessing

Download, normalize, and load Server Machine Dataset files.
Reusable functions shared by code/0_verify_setup.py, notebooks, and scripts.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
    "master/ServerMachineDataset"
)

MACHINE_GROUPS: dict[int, list[int]] = {
    1: list(range(1, 9)),   # machine-1-1 through machine-1-8
    2: list(range(1, 10)),  # machine-2-1 through machine-2-9
    3: list(range(1, 12)),  # machine-3-1 through machine-3-11
}

SUBDIRECTORIES = ["train", "test", "test_label", "interpretation_label"]

REFERENCE_MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]


def get_all_machine_names() -> list[str]:
    """Return all 28 machine names in order."""
    names = []
    for group, ids in MACHINE_GROUPS.items():
        for machine_id in ids:
            names.append(f"machine-{group}-{machine_id}")
    return names


def download_smd_dataset(data_dir: Path, force: bool = False) -> None:
    """Download all SMD data files if missing.

    Args:
        data_dir: Output directory for raw SMD files (e.g., data/smd/raw).
        force: Re-download files even if they already exist.
    """
    machines = get_all_machine_names()
    total_files = len(machines) * len(SUBDIRECTORIES)

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("Downloading SMD dataset: %d machines x %d subdirectories = %d files",
                len(machines), len(SUBDIRECTORIES), total_files)

    downloaded = 0
    skipped = 0
    failed = 0

    with tqdm(total=total_files, desc="Downloading SMD") as pbar:
        for subdir in SUBDIRECTORIES:
            for machine in machines:
                filename = f"{machine}.txt"
                url = f"{BASE_URL}/{subdir}/{filename}"
                dest = data_dir / subdir / filename

                if dest.exists() and not force:
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(response.text)
                    downloaded += 1
                except Exception as e:
                    logger.error("Failed to download %s: %s", url, e)
                    failed += 1

                pbar.update(1)

    logger.info("Download complete: %d downloaded, %d skipped, %d failed",
                downloaded, skipped, failed)


def normalize(
    data: np.ndarray,
    min_vals: np.ndarray | None = None,
    max_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature min-max normalization to [0, 1).

    Reference: tranad/preprocess.py, normalize3().

    Args:
        data: shape (N, features)
        min_vals: per-feature minimums (computed from data if None)
        max_vals: per-feature maximums (computed from data if None)

    Returns:
        (normalized_data, min_vals, max_vals)
    """
    if min_vals is None:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-4), min_vals, max_vals


def parse_interpretation_labels(filepath: Path, shape: tuple[int, int]) -> np.ndarray:
    """Parse interpretation label file into binary matrix.

    Format per line: "start-end:dim1,dim2,dim3"
    Positions are 1-indexed. Converts to 0-indexed.

    Returns:
        Binary matrix of shape (n_timesteps, n_features)
    """
    labels = np.zeros(shape, dtype=np.float64)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pos, values = line.split(":")[0], line.split(":")[1].split(",")
            start = int(pos.split("-")[0])
            end = int(pos.split("-")[1])
            dims = [int(i) - 1 for i in values]
            labels[start - 1 : end - 1, dims] = 1
    return labels


def preprocess_machine(
    machine: str, raw_dir: Path, output_dir: Path
) -> dict[str, tuple]:
    """Preprocess a single machine's data: normalize + save .npy files.

    Returns:
        dict with shapes of saved arrays
    """
    train = np.genfromtxt(
        raw_dir / "train" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )
    test = np.genfromtxt(
        raw_dir / "test" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )
    test_label = np.genfromtxt(
        raw_dir / "test_label" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )

    train_norm, min_vals, max_vals = normalize(train)
    test_norm, _, _ = normalize(test, min_vals, max_vals)

    interp_path = raw_dir / "interpretation_label" / f"{machine}.txt"
    interp_labels = parse_interpretation_labels(interp_path, test.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{machine}_train.npy", train_norm)
    np.save(output_dir / f"{machine}_test.npy", test_norm)
    np.save(output_dir / f"{machine}_test_labels.npy", test_label)
    np.save(output_dir / f"{machine}_interp_labels.npy", interp_labels)
    np.save(
        output_dir / f"{machine}_norm_params.npy",
        np.stack([min_vals, max_vals]),
    )

    return {
        "train": train_norm.shape,
        "test": test_norm.shape,
        "test_labels": test_label.shape,
        "interp_labels": interp_labels.shape,
        "norm_params": (2, train.shape[1]),
    }


def discover_machines(raw_dir: Path) -> list[str]:
    """Find all machine names from the train directory."""
    train_dir = raw_dir / "train"
    return sorted(f.stem for f in train_dir.glob("*.txt"))


def load_processed_data(
    data_dir: Path, machine_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed train/test/labels arrays for a machine.

    Returns:
        (train_data, test_data, test_labels, interp_labels)
    """
    train = np.load(data_dir / f"{machine_id}_train.npy")
    test = np.load(data_dir / f"{machine_id}_test.npy")
    test_labels = np.load(data_dir / f"{machine_id}_test_labels.npy")
    interp_labels = np.load(data_dir / f"{machine_id}_interp_labels.npy")
    return train, test, test_labels, interp_labels


# Swat Preprocessing


def preprocess_swat(raw_dir: Path, output_dir: Path) -> dict[str, tuple]:
    """Preprocess SWaT normal/attack CSVs into TranAD-ready numpy arrays."""

    normal = pd.read_csv(raw_dir / "normal.csv")
    merged = pd.read_csv(raw_dir / "merged.csv")

    # Clean column names because SWaT has spaces like " Timestamp", " MV101"
    normal.columns = normal.columns.str.strip()
    merged.columns = merged.columns.str.strip()

    # Remove duplicate rows
    normal = normal.drop_duplicates()

    # Use a manageable subset for training
    normal = normal.iloc[:73045]

    # Reduce test set size
    merged_labels = merged["Normal/Attack"].astype(str).str.strip()

    normal_test = merged[merged_labels == "Normal"].iloc[:100000]
    attack_test = merged[merged_labels == "Attack"]

    merged = pd.concat([normal_test, attack_test], ignore_index=True)

    print(f"Training rows after cleaning: {len(normal):,}")
    print(f"Reduced test rows: {len(merged):,}")

    # Labels: Normal = 0, Attack = 1
    test_labels = (
        merged["Normal/Attack"]
        .astype(str)
        .str.strip()
        .map({"Normal": 0, "Attack": 1})
        .to_numpy(dtype=np.float64)
    )

    # Drop timestamp and label columns
    train = normal.drop(columns=["Timestamp", "Normal/Attack"])
    test = merged.drop(columns=["Timestamp", "Normal/Attack"])

    # Convert sensor columns to numbers
    train = train.apply(pd.to_numeric, errors="coerce")
    test = test.apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    train = train.ffill().bfill()
    test = test.ffill().bfill()

    train = train.to_numpy(dtype=np.float64)
    test = test.to_numpy(dtype=np.float64)

    # Normalize based on normal training data only
    train_norm, min_vals, max_vals = normalize(train)
    test_norm, _, _ = normalize(test, min_vals, max_vals)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "swat_train.npy", train_norm)
    np.save(output_dir / "swat_test.npy", test_norm)
    np.save(output_dir / "swat_test_labels.npy", test_labels)
    np.save(output_dir / "swat_norm_params.npy", np.stack([min_vals, max_vals]))


    return {
        "train": train_norm.shape,
        "test": test_norm.shape,
        "test_labels": test_labels.shape,
        "norm_params": (2, train.shape[1]),
    }


def load_processed_swat_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed SWaT arrays."""
    train = np.load(data_dir / "swat_train.npy")
    test = np.load(data_dir / "swat_test.npy")
    test_labels = np.load(data_dir / "swat_test_labels.npy")
    return train, test, test_labels

