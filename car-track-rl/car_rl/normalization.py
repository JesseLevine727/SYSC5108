from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == len(self.mean.shape):
            array = array.reshape(1, *array.shape)
        batch_mean = np.mean(array, axis=0)
        batch_var = np.var(array, axis=0)
        batch_count = int(array.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: np.ndarray, clip: float = 5.0) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        normalized = (array - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.astype(np.float32),
            "var": self.var.astype(np.float32),
            "count": float(self.count),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.mean = np.asarray(state_dict["mean"], dtype=np.float64)
        self.var = np.asarray(state_dict["var"], dtype=np.float64)
        self.count = float(state_dict["count"])

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        if batch_count <= 0:
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + (delta * batch_count / total_count)
        current_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        adjustment = np.square(delta) * self.count * batch_count / total_count
        new_var = (current_m2 + batch_m2 + adjustment) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count
