"""Dataset loading, cleaning, label handling, and leakage-safe preprocessing.

The functions in this module are intentionally explicit because the paper must
be reproducible. Scaling is always fitted on training data only, labels are
removed before feature processing, and original attack labels are preserved for
later error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
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
    """Leakage-safe train, validation, and test representation."""

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
    data_quality_report: pd.DataFrame | None = None
    X_validation: pd.DataFrame | None = None
    y_validation: np.ndarray | None = None
    validation_original_labels: pd.Series | None = None
    validation_attack_categories: pd.Series | None = None

    @property
    def has_validation(self) -> bool:
        return (
            self.X_validation is not None
            and self.y_validation is not None
            and self.validation_original_labels is not None
        )


def normalize_label(label: object) -> str:
    """Normalize a dataset label for stable comparisons."""

    return str(label).strip().lower().rstrip(".")


def to_binary_label(label: object, normal_label: str = "normal") -> int:
    """Convert a label to 0 for normal traffic and 1 for attack traffic."""

    return 0 if normalize_label(label) == normalize_label(normal_label) else 1


def map_nsl_kdd_attack_category(label: object) -> str:
    """Map an NSL-KDD label to a standard attack family.

    Some redistributed NSL-KDD test files contain only ``Normal`` and
    ``Attack`` labels. The latter is kept as ``Attack (Unspecified)`` so the
    analysis does not invent a DoS, Probe, R2L, or U2R family that is absent
    from the supplied data.
    """

    cleaned = normalize_label(label)
    if cleaned == "normal":
        return "Normal"
    if cleaned == "attack":
        return "Attack (Unspecified)"
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
    """Read an NSL-KDD file with either 42 or 43 columns.

    Official files contain a difficulty column. Some binary-labelled mirrors
    omit it. Both schemas are accepted and normalized to the same 43 columns.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"NSL-KDD file not found: {file_path}")

    separator = _detect_separator(file_path)
    frame = pd.read_csv(file_path, header=None, sep=separator, engine="c")
    if frame.shape[1] not in {42, 43}:
        fallback = "\t" if separator == "," else ","
        frame = pd.read_csv(file_path, header=None, sep=fallback, engine="c")

    if frame.shape[1] == 43:
        frame.columns = NSL_KDD_COLUMNS
    elif frame.shape[1] == 42:
        frame.columns = NSL_KDD_FEATURE_COLUMNS + ["label"]
        frame["difficulty"] = np.nan
        frame = frame.loc[:, NSL_KDD_COLUMNS]
    else:
        raise ValueError(
            f"Expected 42 or 43 NSL-KDD columns, got {frame.shape[1]} from {file_path}."
        )
    return frame


def _split_development_indices(
    original_labels: pd.Series,
    y: np.ndarray,
    *,
    validation_size: float,
    random_seed: int,
    stratify_by: str,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    indices = np.arange(len(y))
    stratify_labels, effective, pooled = _safe_stratification_labels(
        original_labels,
        y,
        strategy=stratify_by,
    )
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    return train_idx, validation_idx, effective, pooled


def _encode_and_align_partitions(
    train_features: pd.DataFrame,
    other_partitions: Sequence[pd.DataFrame],
    *,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Fit categorical vocabulary on training data and align other partitions."""

    categorical_columns = list(categorical_columns or [])
    present = [column for column in categorical_columns if column in train_features.columns]
    train_encoded = pd.get_dummies(train_features, columns=present, dtype=float)
    train_encoded = train_encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    aligned: list[pd.DataFrame] = []
    for partition in other_partitions:
        encoded = pd.get_dummies(
            partition,
            columns=[column for column in present if column in partition.columns],
            dtype=float,
        )
        encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        aligned.append(
            encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
            .reset_index(drop=True)
            .astype(float)
        )
    return train_encoded.reset_index(drop=True).astype(float), aligned


def _scale_partitions(
    X_train: pd.DataFrame,
    other_partitions: Sequence[pd.DataFrame],
    *,
    scale: bool,
) -> tuple[pd.DataFrame, list[pd.DataFrame], MinMaxScaler]:
    scaler = MinMaxScaler()
    if scale:
        train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
        )
        others = [
            pd.DataFrame(scaler.transform(partition), columns=X_train.columns)
            for partition in other_partitions
        ]
    else:
        scaler.fit(X_train)
        train_scaled = X_train.reset_index(drop=True).astype(float)
        others = [partition.reset_index(drop=True).astype(float) for partition in other_partitions]
    return train_scaled, others, scaler


def prepare_nsl_kdd(
    train_file: str | Path,
    test_file: str | Path,
    *,
    scale: bool = True,
    validation_size: float | None = None,
    validation_seed: int = 42,
    validation_stratify_by: str = "original_label",
) -> PreparedDataset:
    """Load NSL-KDD with an optional train-only validation partition."""

    train_df = read_nsl_kdd_file(train_file).copy()
    test_df = read_nsl_kdd_file(test_file).copy()

    development_labels = train_df["label"].astype(str).reset_index(drop=True)
    test_original = test_df["label"].astype(str).reset_index(drop=True)
    development_y = development_labels.map(to_binary_label).to_numpy(dtype=np.int8)
    y_test = test_original.map(to_binary_label).to_numpy(dtype=np.int8)

    development_features = train_df.drop(columns=["label", "difficulty"], errors="ignore")
    test_features = test_df.drop(columns=["label", "difficulty"], errors="ignore")

    validation_effective = "disabled"
    validation_rare_pooled = 0
    if validation_size is None:
        train_idx = np.arange(len(train_df))
        validation_idx = np.array([], dtype=int)
    else:
        train_idx, validation_idx, validation_effective, validation_rare_pooled = (
            _split_development_indices(
                development_labels,
                development_y,
                validation_size=float(validation_size),
                random_seed=int(validation_seed),
                stratify_by=validation_stratify_by,
            )
        )

    train_features = development_features.iloc[train_idx].reset_index(drop=True)
    validation_features = (
        development_features.iloc[validation_idx].reset_index(drop=True)
        if len(validation_idx)
        else None
    )
    other_raw = [partition for partition in [validation_features, test_features] if partition is not None]
    X_train_encoded, aligned = _encode_and_align_partitions(
        train_features,
        other_raw,
        categorical_columns=NSL_KDD_CATEGORICAL_COLUMNS,
    )
    if validation_features is None:
        X_validation_encoded = None
        X_test_encoded = aligned[0]
    else:
        X_validation_encoded, X_test_encoded = aligned

    scale_inputs = [partition for partition in [X_validation_encoded, X_test_encoded] if partition is not None]
    X_train, scaled, scaler = _scale_partitions(
        X_train_encoded, scale_inputs, scale=scale
    )
    if X_validation_encoded is None:
        X_validation = None
        X_test = scaled[0]
    else:
        X_validation, X_test = scaled

    y_train = development_y[train_idx]
    train_original = development_labels.iloc[train_idx].reset_index(drop=True)
    train_categories = train_original.map(map_nsl_kdd_attack_category)
    test_categories = test_original.map(map_nsl_kdd_attack_category)

    y_validation = development_y[validation_idx] if len(validation_idx) else None
    validation_original = (
        development_labels.iloc[validation_idx].reset_index(drop=True)
        if len(validation_idx)
        else None
    )
    validation_categories = (
        validation_original.map(map_nsl_kdd_attack_category)
        if validation_original is not None
        else None
    )

    detailed_test_categories = sorted(
        set(test_categories) - {"Normal", "Attack (Unspecified)", "Unknown"}
    )
    category_resolution = "detailed" if detailed_test_categories else "binary_only"

    metadata = {
        "dataset": "NSL-KDD",
        "development_records": int(len(train_df)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_validation)) if X_validation is not None else 0,
        "test_records": int(len(X_test)),
        "encoded_features": int(X_train.shape[1]),
        "train_normal_records": int(np.sum(y_train == 0)),
        "train_attack_records": int(np.sum(y_train == 1)),
        "validation_normal_records": int(np.sum(y_validation == 0)) if y_validation is not None else 0,
        "validation_attack_records": int(np.sum(y_validation == 1)) if y_validation is not None else 0,
        "test_normal_records": int(np.sum(y_test == 0)),
        "test_attack_records": int(np.sum(y_test == 1)),
        "validation_size": validation_size,
        "validation_seed": int(validation_seed),
        "validation_stratification_effective": validation_effective,
        "validation_rare_labels_pooled": int(validation_rare_pooled),
        "test_attack_category_resolution": category_resolution,
    }

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_original_labels=train_original,
        test_original_labels=test_original,
        train_attack_categories=train_categories,
        test_attack_categories=test_categories,
        scaler=scaler,
        metadata=metadata,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_original_labels=validation_original,
        validation_attack_categories=validation_categories,
    )


def encode_and_align_train_test(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit categorical columns on training data and align test features safely."""

    train_encoded, aligned = _encode_and_align_partitions(
        train_features,
        [test_features],
        categorical_columns=categorical_columns,
    )
    return train_encoded, aligned[0]


CICIDS2017_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DoS/DDoS", ("ddos", "dos hulk", "dos goldeneye", "dos slowloris", "dos slowhttptest")),
    ("Brute Force", ("ftp-patator", "ssh-patator", "brute force")),
    ("Web Attack", ("web attack", "sql injection", "xss")),
    ("PortScan", ("portscan",)),
    ("Bot", ("bot",)),
    ("Infiltration", ("infiltration", "infilteration")),
    ("Heartbleed", ("heartbleed",)),
)


@dataclass(frozen=True)
class CICIDS2017Source:
    """One CSV source, either an extracted file or a member inside an archive."""

    display_name: str
    path: Path
    archive_member: str | None = None


@dataclass
class _PrioritySampler:
    """Keep a deterministic, memory-bounded sample from streamed records."""

    strategy: str
    benign_label: str
    max_benign_records: int
    max_records_per_attack_label: int
    max_total_records: int | None
    random_seed: int

    def __post_init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}

    def _quota(self, label: str) -> int:
        if self.strategy == "global_cap":
            if self.max_total_records is None:
                raise ValueError("max_total_records is required for global_cap sampling.")
            return self.max_total_records
        if normalize_cicids2017_label(label) == normalize_cicids2017_label(self.benign_label):
            return self.max_benign_records
        return self.max_records_per_attack_label

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        groups = [("__global__", frame)] if self.strategy == "global_cap" else frame.groupby("__original_label__", sort=False)
        for label, group in groups:
            quota = self._quota(str(label))
            existing = self._frames.get(str(label))
            combined = group if existing is None else pd.concat([existing, group], ignore_index=True)
            if len(combined) > quota:
                priorities = combined["__priority__"].to_numpy(dtype=np.uint64, copy=False)
                # Sorting the narrow priority vector avoids pandas copying the
                # full wide CICIDS2017 frame during every streamed chunk.
                selected = np.argsort(priorities, kind="stable")[:quota]
                combined = combined.iloc[selected]
            self._frames[str(label)] = combined.reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame()
        result = pd.concat(self._frames.values(), ignore_index=True)
        return result.sort_values("__priority__", kind="stable").reset_index(drop=True)


def normalize_cicids2017_label(label: object) -> str:
    """Normalize CICIDS2017 labels while preserving their human-readable names."""

    text = str(label).replace("\ufffd", "-").replace("\u0096", "-")
    return " ".join(text.strip().split())


def map_cicids2017_attack_category(label: object, benign_label: str = "BENIGN") -> str:
    """Map original CICIDS2017 labels to stable paper-facing attack families."""

    cleaned = normalize_cicids2017_label(label)
    lowered = cleaned.lower()
    if lowered == normalize_cicids2017_label(benign_label).lower():
        return "Normal"
    for category, patterns in CICIDS2017_CATEGORY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return category
    return "Unknown"


def _deduplicate_column_names(columns: Sequence[object]) -> list[str]:
    """Strip whitespace and make duplicate column names explicit and stable."""

    counts: dict[str, int] = {}
    cleaned: list[str] = []
    for raw in columns:
        base = " ".join(str(raw).replace("\ufeff", "").strip().split())
        counts[base] = counts.get(base, 0) + 1
        cleaned.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return cleaned


def discover_cicids2017_sources(
    raw_dir: str | Path,
    *,
    archive_file: str | Path | None = None,
) -> list[CICIDS2017Source]:
    """Discover official CICIDS2017 machine-learning CSV sources.

    The runner accepts either the official MachineLearningCSV.zip archive or an
    extracted directory. Archive members are streamed directly, so extraction
    is optional and no large temporary copy is required.
    """

    raw_path = Path(raw_dir)
    archive_path = Path(archive_file) if archive_file else raw_path / "MachineLearningCSV.zip"

    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            )
        if not members:
            raise ValueError(f"No CSV files were found inside {archive_path}.")
        return [
            CICIDS2017Source(display_name=Path(member).name, path=archive_path, archive_member=member)
            for member in members
        ]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"CICIDS2017 input not found. Expected archive {archive_path} or directory {raw_path}."
        )

    csv_files = sorted(path for path in raw_path.rglob("*.csv") if path.is_file())
    if not csv_files:
        raise ValueError(f"No CICIDS2017 CSV files were found under {raw_path}.")
    return [CICIDS2017Source(display_name=path.name, path=path) for path in csv_files]


def _iter_cicids2017_chunks(
    source: CICIDS2017Source,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> Iterator[pd.DataFrame]:
    read_options = {
        "chunksize": chunk_size,
        "low_memory": False,
        "encoding_errors": "replace",
    }

    if source.archive_member is None:
        reader = pd.read_csv(source.path, **read_options)
        for index, chunk in enumerate(reader):
            if max_chunks is not None and index >= max_chunks:
                break
            yield chunk
        return

    with zipfile.ZipFile(source.path) as archive:
        with archive.open(source.archive_member) as handle:
            reader = pd.read_csv(handle, **read_options)
            for index, chunk in enumerate(reader):
                if max_chunks is not None and index >= max_chunks:
                    break
                yield chunk


def _clean_cicids2017_chunk(
    chunk: pd.DataFrame,
    *,
    label_column: str,
    benign_label: str,
    source_name: str,
    row_offset: int,
    seen_hashes: set[int],
    random_seed: int,
    drop_duplicates: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one chunk without fitting statistics from future test records."""

    cleaned = chunk.copy()
    cleaned.columns = _deduplicate_column_names(cleaned.columns)
    matching = [name for name in cleaned.columns if name.lower() == label_column.strip().lower()]
    if not matching:
        raise ValueError(f"Label column '{label_column}' was not found in {source_name}.")
    resolved_label = matching[0]

    stats = {
        "rows_read": int(len(cleaned)),
        "rows_missing_label_dropped": 0,
        "rows_all_features_missing_dropped": 0,
        "duplicate_rows_removed": 0,
        "nonfinite_values_replaced": 0,
        "nonnumeric_values_coerced": 0,
        "rows_after_cleaning": 0,
    }

    raw_labels = cleaned[resolved_label]
    valid_label_mask = raw_labels.notna() & raw_labels.astype(str).str.strip().ne("")
    stats["rows_missing_label_dropped"] = int((~valid_label_mask).sum())
    cleaned = cleaned.loc[valid_label_mask].copy()
    raw_labels = cleaned[resolved_label].map(normalize_cicids2017_label)

    features_raw = cleaned.drop(columns=[resolved_label], errors="ignore")
    numeric_columns: dict[str, pd.Series] = {}
    nonnumeric_values_coerced = 0
    for name in features_raw.columns:
        raw_column = features_raw[name]
        converted = pd.to_numeric(raw_column, errors="coerce")
        numeric_columns[name] = converted
        if not is_numeric_dtype(raw_column.dtype):
            meaningful = raw_column.notna() & raw_column.astype(str).str.strip().ne("")
            nonnumeric_values_coerced += int((meaningful & converted.isna()).sum())
    numeric = pd.DataFrame(numeric_columns, index=features_raw.index)
    stats["nonnumeric_values_coerced"] = nonnumeric_values_coerced

    numeric_values = numeric.to_numpy(dtype=float, copy=False)
    nonfinite_mask = np.isinf(numeric_values)
    stats["nonfinite_values_replaced"] = int(nonfinite_mask.sum())
    if stats["nonfinite_values_replaced"]:
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

    valid_feature_mask = ~numeric.isna().all(axis=1)
    stats["rows_all_features_missing_dropped"] = int((~valid_feature_mask).sum())
    numeric = numeric.loc[valid_feature_mask].reset_index(drop=True)
    labels = raw_labels.loc[valid_feature_mask].reset_index(drop=True)

    hash_frame = numeric.copy()
    hash_frame["__label__"] = labels
    row_hashes = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(dtype=np.uint64)

    if drop_duplicates:
        keep = np.ones(len(row_hashes), dtype=bool)
        for idx, value in enumerate(row_hashes):
            key = int(value)
            if key in seen_hashes:
                keep[idx] = False
            else:
                seen_hashes.add(key)
        stats["duplicate_rows_removed"] = int((~keep).sum())
        numeric = numeric.loc[keep].reset_index(drop=True)
        labels = labels.loc[keep].reset_index(drop=True)
        row_hashes = row_hashes[keep]

    y = labels.map(lambda value: 0 if value.upper() == benign_label.upper() else 1).astype(np.int8)
    categories = labels.map(lambda value: map_cicids2017_attack_category(value, benign_label))
    source_rows = np.arange(row_offset, row_offset + len(labels), dtype=np.int64)
    seed_mix = np.uint64((int(random_seed) * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF)
    priorities = row_hashes ^ seed_mix

    result = numeric.reset_index(drop=True)
    result["__binary_label__"] = y.to_numpy()
    result["__original_label__"] = labels.to_numpy()
    result["__attack_category__"] = categories.to_numpy()
    result["__source_file__"] = source_name
    result["__source_row__"] = source_rows
    result["__priority__"] = priorities
    stats["rows_after_cleaning"] = int(len(result))
    return result, stats


def clean_cicids2017_dataframe(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Clean one in-memory CICIDS2017 dataframe for tests and small studies."""

    frame, _stats = _clean_cicids2017_chunk(
        df,
        label_column=label_column,
        benign_label=benign_label,
        source_name="in_memory",
        row_offset=0,
        seen_hashes=set(),
        random_seed=42,
        drop_duplicates=drop_duplicates,
    )
    helper_columns = [name for name in frame.columns if name.startswith("__")]
    X = frame.drop(columns=helper_columns).copy()
    y = frame["__binary_label__"].to_numpy(dtype=np.int8)
    original = frame["__original_label__"].astype(str).reset_index(drop=True)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0).astype(float).reset_index(drop=True)
    return X, y, original


def load_cicids2017_csv_files(
    files: Iterable[str | Path],
    *,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load a small collection of extracted CICIDS2017 CSV files."""

    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    original_labels: list[pd.Series] = []
    for file in files:
        raw = pd.read_csv(file, low_memory=False)
        X_part, y_part, original_part = clean_cicids2017_dataframe(
            raw,
            label_column=label_column,
            benign_label=benign_label,
        )
        frames.append(X_part)
        labels.append(y_part)
        original_labels.append(original_part)

    if not frames:
        raise ValueError("No CICIDS2017 files were provided.")

    columns = frames[0].columns
    aligned = [frame.reindex(columns=columns, fill_value=0.0) for frame in frames]
    return (
        pd.concat(aligned, ignore_index=True).astype(float),
        np.concatenate(labels).astype(np.int8),
        pd.concat(original_labels, ignore_index=True),
    )


def _safe_stratification_labels(
    original_labels: pd.Series,
    y: np.ndarray,
    *,
    strategy: str,
) -> tuple[np.ndarray, str, int]:
    """Build stable split strata without discarding common attack labels.

    CICIDS2017 contains extremely rare labels after exact-duplicate removal.
    A single one-record label prevents direct multiclass stratification. Rather
    than falling back to binary stratification for the entire dataset, rare
    labels are pooled by binary class while common labels retain their original
    strata. If even the pooled strata are too small, binary stratification is
    used as the final safe fallback.
    """

    labels = original_labels.astype(str).reset_index(drop=True)
    if strategy == "binary":
        return np.asarray(y, dtype=np.int8), "binary", 0

    counts = labels.value_counts()
    rare_labels = set(counts[counts < 2].index.astype(str))
    if not rare_labels:
        return labels.to_numpy(), "original_label", 0

    pooled = labels.copy()
    rare_mask = pooled.isin(rare_labels)
    binary_labels = np.asarray(y, dtype=np.int8)
    pooled.loc[rare_mask & (binary_labels == 0)] = "__RARE_NORMAL__"
    pooled.loc[rare_mask & (binary_labels == 1)] = "__RARE_ATTACK__"
    pooled_counts = pooled.value_counts()
    if len(pooled_counts) > 1 and int(pooled_counts.min()) >= 2:
        return (
            pooled.to_numpy(),
            "original_label_with_rare_pooling",
            len(rare_labels),
        )

    return binary_labels, "binary_fallback", len(rare_labels)


def prepare_cicids2017(
    raw_dir: str | Path,
    *,
    archive_file: str | Path | None = None,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
    chunk_size: int = 25_000,
    max_chunks_per_file: int | None = None,
    drop_duplicates: bool = True,
    sampling_strategy: str = "per_label_cap",
    max_benign_records: int = 20_000,
    max_records_per_attack_label: int = 5_000,
    max_total_records: int | None = None,
    sampling_seed: int = 42,
    test_size: float = 0.30,
    split_seed: int = 42,
    stratify_by: str = "original_label",
    scale: bool = True,
    validation_size: float | None = None,
    validation_seed: int = 42,
) -> PreparedDataset:
    """Prepare CICIDS2017 with train-only fitting and optional validation.

    Exact duplicates are removed before splitting. The final test partition is
    created first and remains untouched. When validation is enabled, only the
    development partition is split again. Imputation and scaling are fitted on
    the detector-training partition and applied unchanged to validation and
    test data.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_chunks_per_file is not None and max_chunks_per_file <= 0:
        raise ValueError("max_chunks_per_file must be positive or null.")
    if sampling_strategy not in {"per_label_cap", "global_cap"}:
        raise ValueError("sampling_strategy must be 'per_label_cap' or 'global_cap'.")
    if max_benign_records <= 0 or max_records_per_attack_label <= 0:
        raise ValueError("Per-label sampling quotas must be positive.")
    if sampling_strategy == "global_cap" and (
        max_total_records is None or max_total_records <= 0
    ):
        raise ValueError("max_total_records must be positive for global_cap sampling.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size is not None and not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if stratify_by not in {"original_label", "binary"}:
        raise ValueError("stratify_by must be 'original_label' or 'binary'.")

    sources = discover_cicids2017_sources(raw_dir, archive_file=archive_file)
    sampler = _PrioritySampler(
        strategy=sampling_strategy,
        benign_label=benign_label,
        max_benign_records=max_benign_records,
        max_records_per_attack_label=max_records_per_attack_label,
        max_total_records=max_total_records,
        random_seed=sampling_seed,
    )
    seen_hashes: set[int] = set()
    report_rows: list[dict[str, object]] = []

    for source in sources:
        source_stats: dict[str, object] = {
            "record_type": "source",
            "source_file": source.display_name,
            "archive_member": source.archive_member or "",
            "scan_complete": max_chunks_per_file is None,
            "chunks_read": 0,
            "rows_read": 0,
            "rows_missing_label_dropped": 0,
            "rows_all_features_missing_dropped": 0,
            "duplicate_rows_removed": 0,
            "nonfinite_values_replaced": 0,
            "nonnumeric_values_coerced": 0,
            "rows_after_cleaning": 0,
            "rows_retained_in_sample": 0,
        }
        row_offset = 0
        for chunk in _iter_cicids2017_chunks(
            source,
            chunk_size=chunk_size,
            max_chunks=max_chunks_per_file,
        ):
            cleaned, chunk_stats = _clean_cicids2017_chunk(
                chunk,
                label_column=label_column,
                benign_label=benign_label,
                source_name=source.display_name,
                row_offset=row_offset,
                seen_hashes=seen_hashes,
                random_seed=sampling_seed,
                drop_duplicates=drop_duplicates,
            )
            sampler.add(cleaned)
            source_stats["chunks_read"] = int(source_stats["chunks_read"]) + 1
            for key, value in chunk_stats.items():
                source_stats[key] = int(source_stats[key]) + int(value)
            row_offset += len(chunk)
        report_rows.append(source_stats)

    sampled = sampler.build()
    if sampled.empty:
        raise ValueError("CICIDS2017 preprocessing produced no usable records.")

    retained_by_source = sampled["__source_file__"].value_counts()
    for row in report_rows:
        row["rows_retained_in_sample"] = int(retained_by_source.get(row["source_file"], 0))

    helper_columns = [name for name in sampled.columns if name.startswith("__")]
    feature_columns = [name for name in sampled.columns if name not in helper_columns]
    X_all = sampled[feature_columns].copy()
    y_all = sampled["__binary_label__"].to_numpy(dtype=np.int8)
    original_all = sampled["__original_label__"].astype(str).reset_index(drop=True)
    categories_all = sampled["__attack_category__"].astype(str).reset_index(drop=True)

    all_indices = np.arange(len(sampled))
    test_stratify, test_stratification, test_rare_pooled = _safe_stratification_labels(
        original_all,
        y_all,
        strategy=stratify_by,
    )
    development_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=split_seed,
        shuffle=True,
        stratify=test_stratify,
    )

    validation_stratification = "disabled"
    validation_rare_pooled = 0
    if validation_size is None:
        train_idx = np.asarray(development_idx, dtype=int)
        validation_idx = np.array([], dtype=int)
    else:
        development_original = original_all.iloc[development_idx].reset_index(drop=True)
        development_y = y_all[development_idx]
        local_train, local_validation, validation_stratification, validation_rare_pooled = (
            _split_development_indices(
                development_original,
                development_y,
                validation_size=float(validation_size),
                random_seed=int(validation_seed),
                stratify_by=stratify_by,
            )
        )
        train_idx = np.asarray(development_idx, dtype=int)[local_train]
        validation_idx = np.asarray(development_idx, dtype=int)[local_validation]

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_validation_raw = (
        X_all.iloc[validation_idx].reset_index(drop=True) if len(validation_idx) else None
    )
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

    y_train = y_all[train_idx]
    y_validation = y_all[validation_idx] if len(validation_idx) else None
    y_test = y_all[test_idx]
    train_original = original_all.iloc[train_idx].reset_index(drop=True)
    validation_original = (
        original_all.iloc[validation_idx].reset_index(drop=True)
        if len(validation_idx)
        else None
    )
    test_original = original_all.iloc[test_idx].reset_index(drop=True)
    train_categories = categories_all.iloc[train_idx].reset_index(drop=True)
    validation_categories = (
        categories_all.iloc[validation_idx].reset_index(drop=True)
        if len(validation_idx)
        else None
    )
    test_categories = categories_all.iloc[test_idx].reset_index(drop=True)

    train_medians = X_train_raw.median(numeric_only=True)
    all_missing_columns = train_medians[train_medians.isna()].index.tolist()
    if all_missing_columns:
        X_train_raw = X_train_raw.drop(columns=all_missing_columns)
        X_test_raw = X_test_raw.drop(columns=all_missing_columns)
        if X_validation_raw is not None:
            X_validation_raw = X_validation_raw.drop(columns=all_missing_columns)
        train_medians = X_train_raw.median(numeric_only=True)

    train_missing_before = int(X_train_raw.isna().to_numpy().sum())
    validation_missing_before = (
        int(X_validation_raw.isna().to_numpy().sum())
        if X_validation_raw is not None
        else 0
    )
    test_missing_before = int(X_test_raw.isna().to_numpy().sum())
    X_train_imputed = X_train_raw.fillna(train_medians).fillna(0.0).astype(float)
    X_validation_imputed = (
        X_validation_raw.fillna(train_medians).fillna(0.0).astype(float)
        if X_validation_raw is not None
        else None
    )
    X_test_imputed = X_test_raw.fillna(train_medians).fillna(0.0).astype(float)

    other_imputed = [
        partition
        for partition in [X_validation_imputed, X_test_imputed]
        if partition is not None
    ]
    X_train, scaled, scaler = _scale_partitions(
        X_train_imputed,
        other_imputed,
        scale=scale,
    )
    if X_validation_imputed is None:
        X_validation = None
        X_test = scaled[0]
    else:
        X_validation, X_test = scaled

    summary = {
        "record_type": "summary",
        "source_file": "ALL_SOURCES",
        "archive_member": "",
        "scan_complete": all(bool(row["scan_complete"]) for row in report_rows),
        "chunks_read": sum(int(row["chunks_read"]) for row in report_rows),
        "rows_read": sum(int(row["rows_read"]) for row in report_rows),
        "rows_missing_label_dropped": sum(
            int(row["rows_missing_label_dropped"]) for row in report_rows
        ),
        "rows_all_features_missing_dropped": sum(
            int(row["rows_all_features_missing_dropped"]) for row in report_rows
        ),
        "duplicate_rows_removed": sum(
            int(row["duplicate_rows_removed"]) for row in report_rows
        ),
        "nonfinite_values_replaced": sum(
            int(row["nonfinite_values_replaced"]) for row in report_rows
        ),
        "nonnumeric_values_coerced": sum(
            int(row["nonnumeric_values_coerced"]) for row in report_rows
        ),
        "rows_after_cleaning": sum(int(row["rows_after_cleaning"]) for row in report_rows),
        "rows_retained_in_sample": int(len(sampled)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_validation)) if X_validation is not None else 0,
        "test_records": int(len(X_test)),
        "train_missing_values_imputed": train_missing_before,
        "validation_missing_values_imputed": validation_missing_before,
        "test_missing_values_imputed": test_missing_before,
        "all_missing_training_columns_removed": len(all_missing_columns),
        "test_stratification_strategy_effective": test_stratification,
        "validation_stratification_strategy_effective": validation_stratification,
        "test_rare_labels_pooled": int(test_rare_pooled),
        "validation_rare_labels_pooled": int(validation_rare_pooled),
    }

    label_rows: list[dict[str, object]] = []
    for label in sorted(original_all.unique()):
        label_rows.append(
            {
                "record_type": "label_distribution",
                "source_file": "ALL_SOURCES",
                "original_label": label,
                "attack_category": map_cicids2017_attack_category(label, benign_label),
                "binary_label": 0 if label.upper() == benign_label.upper() else 1,
                "rows_retained_in_sample": int((original_all == label).sum()),
                "train_records": int((train_original == label).sum()),
                "validation_records": (
                    int((validation_original == label).sum())
                    if validation_original is not None
                    else 0
                ),
                "test_records": int((test_original == label).sum()),
            }
        )
    quality_report = pd.DataFrame([*report_rows, summary, *label_rows])

    metadata: dict[str, object] = {
        "dataset": "CICIDS2017",
        "source_files": len(sources),
        "scan_complete": bool(summary["scan_complete"]),
        "sampled_records": int(len(sampled)),
        "development_records": int(len(development_idx)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_validation)) if X_validation is not None else 0,
        "test_records": int(len(X_test)),
        "encoded_features": int(X_train.shape[1]),
        "train_normal_records": int(np.sum(y_train == 0)),
        "train_attack_records": int(np.sum(y_train == 1)),
        "validation_normal_records": int(np.sum(y_validation == 0)) if y_validation is not None else 0,
        "validation_attack_records": int(np.sum(y_validation == 1)) if y_validation is not None else 0,
        "test_normal_records": int(np.sum(y_test == 0)),
        "test_attack_records": int(np.sum(y_test == 1)),
        "test_split_strategy": f"stratified_random:{test_stratification}",
        "validation_split_strategy": f"stratified_random:{validation_stratification}",
        "test_rare_labels_pooled": int(test_rare_pooled),
        "validation_rare_labels_pooled": int(validation_rare_pooled),
        "test_size": float(test_size),
        "validation_size": validation_size,
        "sampling_strategy": sampling_strategy,
        "sampling_seed": int(sampling_seed),
        "split_seed": int(split_seed),
        "validation_seed": int(validation_seed),
        "max_benign_records": int(max_benign_records),
        "max_records_per_attack_label": int(max_records_per_attack_label),
        "max_total_records": max_total_records,
        "removed_all_missing_columns": all_missing_columns,
    }

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_original_labels=train_original,
        test_original_labels=test_original,
        train_attack_categories=train_categories,
        test_attack_categories=test_categories,
        scaler=scaler,
        metadata=metadata,
        data_quality_report=quality_report,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_original_labels=validation_original,
        validation_attack_categories=validation_categories,
    )
