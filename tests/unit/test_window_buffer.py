from __future__ import annotations

import numpy as np
import pytest

from app.detection.window_buffer import SlidingWindowBuffer


def _make_values(n_features: int = 14, val: float = 0.5) -> list[float]:
    return [val] * n_features


class TestSlidingWindowBuffer:
    def test_returns_none_while_filling(self) -> None:
        buf = SlidingWindowBuffer(window_size=30, n_features=14)
        for i in range(29):
            result = buf.push(_make_values())
            assert result is None

    def test_returns_window_when_full(self) -> None:
        buf = SlidingWindowBuffer(window_size=30, n_features=14)
        result = None
        for i in range(30):
            result = buf.push(_make_values(val=float(i)))
        assert result is not None
        assert result.shape == (1, 30, 14)

    def test_returns_window_on_every_subsequent_push(self) -> None:
        buf = SlidingWindowBuffer(window_size=5, n_features=3)
        for _ in range(5):
            buf.push([1.0, 2.0, 3.0])
        # 6th push should also return a window (stride=1)
        result = buf.push([4.0, 5.0, 6.0])
        assert result is not None
        assert result.shape == (1, 5, 3)

    def test_deque_slides_correctly(self) -> None:
        buf = SlidingWindowBuffer(window_size=3, n_features=1)
        for v in [1.0, 2.0, 3.0]:
            buf.push([v])
        window = buf.push([4.0])
        # Buffer should now contain [2, 3, 4]
        assert window is not None
        np.testing.assert_array_almost_equal(
            window[0, :, 0], [2.0, 3.0, 4.0]
        )

    def test_wrong_feature_count_raises(self) -> None:
        buf = SlidingWindowBuffer(window_size=5, n_features=3)
        with pytest.raises(ValueError, match="3 feature values"):
            buf.push([1.0, 2.0])

    def test_is_ready_after_warmup(self) -> None:
        buf = SlidingWindowBuffer(window_size=5, n_features=2)
        assert not buf.is_ready()
        for _ in range(5):
            buf.push([0.0, 0.0])
        assert buf.is_ready()

    def test_reset_clears_buffer(self) -> None:
        buf = SlidingWindowBuffer(window_size=5, n_features=2)
        for _ in range(5):
            buf.push([0.0, 0.0])
        buf.reset()
        assert not buf.is_ready()
        assert buf.fill_level == 0

    def test_window_dtype_is_float32(self) -> None:
        buf = SlidingWindowBuffer(window_size=3, n_features=2)
        for _ in range(3):
            buf.push([1.0, 2.0])
        result = buf.push([3.0, 4.0])
        assert result is not None
        assert result.dtype == np.float32
