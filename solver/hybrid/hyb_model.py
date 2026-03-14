from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class HYBModel:
    """Hybrid ensemble: average of RandomForest and SVR predictions.

    Used to rank candidate placements by predicted downstream value.
    """

    def __init__(self, model_dir: Path = Path("models"), alpha: float = 0.5):
        self.model_dir = model_dir
        self.alpha = alpha  # weight for RF, (1-alpha) for SVR
        self.rf = None
        self.svr_pipeline = None
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train RF and SVR on the given data."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        # RF trains on raw features
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
        )
        self.rf.fit(X, y)

        # SVR needs scaled features; subsample if dataset is large
        svr_max_samples = 10000
        if len(X) > svr_max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), svr_max_samples, replace=False)
            X_svr, y_svr = X[idx], y[idx]
        else:
            X_svr, y_svr = X, y

        self.svr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ])
        self.svr_pipeline.fit(X_svr, y_svr)

        self._is_trained = True

        # Return training metrics
        rf_pred = self.rf.predict(X)
        svr_pred = self.svr_pipeline.predict(X)
        hyb_pred = self.alpha * rf_pred + (1 - self.alpha) * svr_pred

        return {
            "rf_mse": float(np.mean((rf_pred - y) ** 2)),
            "svr_mse": float(np.mean((svr_pred - y) ** 2)),
            "hyb_mse": float(np.mean((hyb_pred - y) ** 2)),
            "train_size": len(X),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return averaged scores. If not trained, returns zeros."""
        if not self._is_trained:
            return np.zeros(X.shape[0])

        rf_pred = self.rf.predict(X)
        svr_pred = self.svr_pipeline.predict(X)
        return self.alpha * rf_pred + (1 - self.alpha) * svr_pred

    def rank(self, X: np.ndarray) -> np.ndarray:
        """Return indices sorted by descending predicted score."""
        scores = self.predict(X)
        return np.argsort(-scores)

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_dir / "rf.pkl", "wb") as f:
            pickle.dump(self.rf, f)
        with open(self.model_dir / "svr_pipeline.pkl", "wb") as f:
            pickle.dump(self.svr_pipeline, f)
        with open(self.model_dir / "meta.pkl", "wb") as f:
            pickle.dump({"alpha": self.alpha}, f)

    def load(self) -> bool:
        """Return True if loaded successfully."""
        rf_path = self.model_dir / "rf.pkl"
        svr_path = self.model_dir / "svr_pipeline.pkl"
        meta_path = self.model_dir / "meta.pkl"

        if not rf_path.exists() or not svr_path.exists():
            return False

        try:
            with open(rf_path, "rb") as f:
                self.rf = pickle.load(f)
            with open(svr_path, "rb") as f:
                self.svr_pipeline = pickle.load(f)
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.alpha = meta.get("alpha", 0.5)
            self._is_trained = True
            return True
        except Exception:
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained
