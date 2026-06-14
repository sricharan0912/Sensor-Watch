from __future__ import annotations

from collections import deque

import numpy as np


class SlidingWindowBuffer:
    """Per-engine circular buffer that yields a normalised (1, W, F) window when full.

    One instance lives per engine_id in app.state.window_buffers.
    At stride=1, every new reading after warmup triggers inference.

    Multi-worker note: this is process-local. For multi-worker deployments
    the deque must be moved to a Redis list — the interface is unchanged.
    """

    def __init__(self, window_size: int = 30, n_features: int = 14) -> None:
        self._window_size = window_size
        self._n_features = n_features
        self._buf: deque[list[float]] = deque(maxlen=window_size)

    def push(self, values: list[float]) -> np.ndarray | None:
        """Append *values* (length n_features) to the buffer.

        Returns shape (1, window_size, n_features) ndarray when the buffer is
        full, otherwise None.
        """
        if len(values) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature values, got {len(values)}"
            )
        self._buf.append(values)
        if len(self._buf) == self._window_size:
            window = np.array(list(self._buf), dtype=np.float32)  # (W, F)
            return window[np.newaxis, ...]  # (1, W, F)
        return None

    def is_ready(self) -> bool:
        return len(self._buf) == self._window_size

    def reset(self) -> None:
        self._buf.clear()

    @property
    def fill_level(self) -> int:
        return len(self._buf)
