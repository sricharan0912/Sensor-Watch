from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.preprocessor import fit_scaler, sliding_windows, transform


def _make_df(n_rows: int = 100, n_features: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = [f"s{i}" for i in range(1, n_features + 1)]
    data = rng.uniform(100.0, 500.0, size=(n_rows, n_features))
    return pd.DataFrame(data, columns=cols)


class TestFitScaler:
    def test_transform_output_in_unit_range(self) -> None:
        cols = ["s1", "s2", "s3"]
        df = _make_df(100, 3)
        scaler = fit_scaler(df, cols)
        result = transform(df, scaler, cols)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_transform_shape(self) -> None:
        cols = ["s1", "s2", "s3"]
        df = _make_df(50, 3)
        scaler = fit_scaler(df, cols)
        result = transform(df, scaler, cols)
        assert result.shape == (50, 3)

    def test_dtype_is_float32(self) -> None:
        cols = ["s1"]
        df = _make_df(10, 1)
        scaler = fit_scaler(df, cols)
        result = transform(df, scaler, cols)
        assert result.dtype == np.float32


class TestSlidingWindows:
    def test_basic_shape(self) -> None:
        arr = np.zeros((100, 14), dtype=np.float32)
        windows = sliding_windows(arr, window_size=30, stride=1)
        assert windows.shape == (71, 30, 14)  # (100 - 30 + 1) = 71

    def test_stride_2(self) -> None:
        arr = np.zeros((100, 5), dtype=np.float32)
        windows = sliding_windows(arr, window_size=10, stride=2)
        # (100 - 10) // 2 + 1 = 46
        assert windows.shape == (46, 10, 5)

    def test_empty_when_too_short(self) -> None:
        arr = np.zeros((10, 3), dtype=np.float32)
        windows = sliding_windows(arr, window_size=30)
        assert windows.shape[0] == 0

    def test_window_values_are_correct(self) -> None:
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)
        windows = sliding_windows(arr, window_size=3, stride=1)
        # First window: [0, 1, 2]
        np.testing.assert_array_equal(windows[0, :, 0], [0.0, 1.0, 2.0])
        # Second window: [1, 2, 3]
        np.testing.assert_array_equal(windows[1, :, 0], [1.0, 2.0, 3.0])

    def test_windows_are_independent_copies(self) -> None:
        arr = np.ones((50, 2), dtype=np.float32)
        windows = sliding_windows(arr, window_size=5)
        # Modifying a window should not affect the original array
        windows[0, 0, 0] = 999.0
        assert arr[0, 0] == 1.0
