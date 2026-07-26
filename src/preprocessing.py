"""Dataset loading, cleaning, label handling, and leakage-safe preprocessing.

The functions in this module are intentionally explicit because the paper must
be reproducible. Scaling is always fitted on training data only, labels are
removed before feature processing, and original attack labels are preserved for
later error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


NSL_KDD_FEATURE_COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_COLUMNS: list[str] = NSL_KDD_FEATURE_COLUMNS + ["label", "difficulty"]
NSL_KDD_CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]

NSL_KDD_DOS_ATTACKS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "apache2",
    "udpstorm",
    "processtable",
    "mailbomb",
    "worm",
}

NSL_KDD_PROBE_ATTACKS = {
    "ipsweep",
    "nmap",
    "portsweep",
    "satan",
    "mscan",
    "saint",
}

NSL_KDD_R2L_ATTACKS = {
    "ftp_write",
    "guess_passwd",
    "imap",
    "multihop",
    "phf",
    "spy",
    "warezclient",
    "warezmaster",
    "sendmail",
    "named",
    "snmpgetattack",
    "snmpguess",
    "xlock",
    "xsnoop",
    "httptunnel",
}

NSL_KDD_U2R_ATTACKS = {
    "buffer_overflow",
    "loadmodule",
    "perl",
    "rootkit",
    "ps",
    "sqlattack",
    "xterm",
}


@dataclass(frozen=True)
class PreparedDataset:
    """Leakage-safe train/test representation used by experiments."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_original_labels: pd.Series
    test_original_labels: pd.Series
    train_attack_categories: pd.Series | None
    test_attack_categories: pd.Series | None
    scaler: MinMaxScaler
    metadata: dict[str, object]


def normalize_label(label: object) -> str:
    """Normalize a dataset label for stable comparisons."""

    return str(label).strip().lower()


def to_binary_label(label: object, normal_label: str = "normal") -> int:
    """Convert a label to 0 for normal traffic and 1 for attack traffic."""

    return 0 if normalize_label(label) == normalize_label(normal_label) else 1


def map_nsl_kdd_attack_category(label: object) -> str:
    """Map an NSL-KDD label to Normal, DoS, Probe, R2L, U2R, or Unknown."""

    cleaned = normalize_label(label)
    if cleaned == "normal":
        return "Normal"
    if cleaned in NSL_KDD_DOS_ATTACKS:
        return "DoS"
    if cleaned in NSL_KDD_PROBE_ATTACKS:
        return "Probe"
    if cleaned in NSL_KDD_R2L_ATTACKS:
        return "R2L"
    if cleaned in NSL_KDD_U2R_ATTACKS:
        return "U2R"
    return "Unknown"


def _detect_separator(path: str | Path) -> str:
    """Detect whether an NSL-KDD file is comma separated or tab separated."""

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    if "\t" in first_line and first_line.count("\t") >= first_line.count(","):
        return "\t"
    return ","


def read_nsl_kdd_file(path: str | Path) -> pd.DataFrame:
    """Read one NSL-KDD train or test file.

    The function supports the common comma-separated files and tab-separated
    mirrors. It raises a clear error if the file does not resolve to 43 columns.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"NSL-KDD file not found: {file_path}")

    separator = _detect_separator(file_path)
    df = pd.read_csv(file_path, names=NSL_KDD_COLUMNS, sep=separator, engine="c")

    if df.shape[1] != len(NSL_KDD_COLUMNS):
        fallback = "\t" if separator == "," else ","
        df = pd.read_csv(file_path, names=NSL_KDD_COLUMNS, sep=fallback, engine="c")

    if df.shape[1] != len(NSL_KDD_COLUMNS):
        raise ValueError(
            f"Expected {len(NSL_KDD_COLUMNS)} NSL-KDD columns, got {df.shape[1]} from {file_path}."
        )

    return df


def prepare_nsl_kdd(
    train_file: str | Path,
    test_file: str | Path,
    *,
    scale: bool = True,
) -> PreparedDataset:
    """Load and preprocess NSL-KDD using train-only scaling.

    Output features are numeric and aligned across train and test. Original
    labels and attack categories are preserved outside the feature matrix.
    """

    train_df = read_nsl_kdd_file(train_file).copy()
    test_df = read_nsl_kdd_file(test_file).copy()

    train_original_labels = train_df["label"].astype(str).copy()
    test_original_labels = test_df["label"].astype(str).copy()

    y_train = train_original_labels.apply(to_binary_label).to_numpy(dtype=np.int8)
    y_test = test_original_labels.apply(to_binary_label).to_numpy(dtype=np.int8)

    train_attack_categories = train_original_labels.apply(map_nsl_kdd_attack_category)
    test_attack_categories = test_original_labels.apply(map_nsl_kdd_attack_category)

    train_features = train_df.drop(columns=["label", "difficulty"], errors="ignore")
    test_features = test_df.drop(columns=["label", "difficulty"], errors="ignore")

    X_train, X_test = encode_and_align_train_test(
        train_features,
        test_features,
        categorical_columns=NSL_KDD_CATEGORICAL_COLUMNS,
    )

    scaler = MinMaxScaler()
    if scale:
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    else:
        scaler.fit(X_train)
        X_train_scaled = X_train.astype(float)
        X_test_scaled = X_test.astype(float)

    metadata = {
        "dataset": "NSL-KDD",
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "encoded_features": int(X_train_scaled.shape[1]),
        "train_normal_records": int(np.sum(y_train == 0)),
        "train_attack_records": int(np.sum(y_train == 1)),
        "test_normal_records": int(np.sum(y_test == 0)),
        "test_attack_records": int(np.sum(y_test == 1)),
    }

    return PreparedDataset(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        train_original_labels=train_original_labels,
        test_original_labels=test_original_labels,
        train_attack_categories=train_attack_categories,
        test_attack_categories=test_attack_categories,
        scaler=scaler,
        metadata=metadata,
    )


def encode_and_align_train_test(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit categorical columns on training data and align test features safely.

    Test-only categories are not added to the learned feature space. Their
    categorical indicators remain zero, which avoids transductive test-set
    information entering the training representation.
    """

    categorical_columns = list(categorical_columns or [])
    present_categoricals = [col for col in categorical_columns if col in train_features.columns]

    train_encoded = pd.get_dummies(
        train_features,
        columns=present_categoricals,
        dtype=float,
    )
    test_encoded = pd.get_dummies(
        test_features,
        columns=[col for col in present_categoricals if col in test_features.columns],
        dtype=float,
    )

    train_encoded = train_encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_encoded = test_encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_aligned = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)

    return (
        train_encoded.reset_index(drop=True).astype(float),
        test_aligned.reset_index(drop=True).astype(float),
    )


def clean_cicids2017_dataframe(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Clean one CICIDS2017 dataframe for binary experiments.

    This helper prepares a single dataframe. A future runner can decide whether
    to split by file/day or use a stratified train/test split.
    """

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if label_column not in cleaned.columns:
        matching = [col for col in cleaned.columns if col.lower() == label_column.lower()]
        if not matching:
            raise ValueError(f"Label column not found: {label_column}")
        label_column = matching[0]

    cleaned = cleaned.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)
    cleaned = cleaned.dropna(subset=[label_column])

    original_labels = cleaned[label_column].astype(str).str.strip().copy()
    y = original_labels.apply(lambda value: 0 if value.upper() == benign_label.upper() else 1).to_numpy(dtype=np.int8)

    X = cleaned.drop(columns=[label_column], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    if drop_duplicates:
        before = len(X)
        combined = X.copy()
        combined["__label__"] = y
        combined["__original_label__"] = original_labels.to_numpy()
        combined = combined.drop_duplicates().reset_index(drop=True)
        original_labels = combined["__original_label__"].copy()
        y = combined["__label__"].to_numpy(dtype=np.int8)
        X = combined.drop(columns=["__label__", "__original_label__"])
        if len(X) == 0 and before > 0:
            raise ValueError("CICIDS2017 cleaning removed all rows.")

    return X.astype(float).reset_index(drop=True), y, original_labels.reset_index(drop=True)


def load_cicids2017_csv_files(
    files: Iterable[str | Path],
    *,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load and clean multiple CICIDS2017 CSV files as one dataset."""

    frames = []
    labels = []
    original_labels = []

    for file in files:
        raw = pd.read_csv(file)
        X_part, y_part, label_part = clean_cicids2017_dataframe(
            raw,
            label_column=label_column,
            benign_label=benign_label,
        )
        frames.append(X_part)
        labels.append(y_part)
        original_labels.append(label_part)

    if not frames:
        raise ValueError("No CICIDS2017 files were provided.")

    X = pd.concat(frames, axis=0, ignore_index=True).fillna(0.0)
    y = np.concatenate(labels).astype(np.int8)
    original = pd.concat(original_labels, axis=0, ignore_index=True)

    return X, y, original
