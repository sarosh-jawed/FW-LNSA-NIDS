"""Complete persisted raw-row transform plus frozen binary detector inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..preprocessing import NSL_KDD_CATEGORICAL_COLUMNS
from ..confirmatory_protocol import predict_from_native_scores
from ..representation import transform_with_medians


@dataclass
class DeploymentPipeline:
    model: object
    scaler: object
    encoded_columns: list[str]
    selected_features: list[str]
    binary_medians: pd.Series
    imputation_medians: pd.Series | None
    dataset_key: str

    def transform(self, frame):
        X = frame.copy()
        if self.dataset_key == "nsl_kdd":
            X = X.drop(columns=["label", "difficulty"], errors="ignore")
            X = pd.get_dummies(
                X,
                columns=[c for c in NSL_KDD_CATEGORICAL_COLUMNS if c in X],
                dtype=float,
            )
            X = X.reindex(columns=self.encoded_columns, fill_value=0.0).astype(float)
        elif self.dataset_key == "cicids2017":
            X.columns = [" ".join(str(c).strip().split()) for c in X.columns]
            X = (
                X.reindex(columns=self.encoded_columns)
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            if self.imputation_medians is None:
                raise ValueError("Missing training imputation state")
            X = X.fillna(self.imputation_medians).fillna(0.0)
        else:
            raise ValueError(
                "Deployment raw transform supports only the two documented datasets"
            )
        X = pd.DataFrame(self.scaler.transform(X), columns=self.encoded_columns)
        return transform_with_medians(X[self.selected_features], self.binary_medians)

    def predict(self, frame):
        return predict_from_native_scores(
            self.model.decision_scores(self.transform(frame)),
            method=self.model.method,
            threshold=self.model.detection_threshold,
        )
