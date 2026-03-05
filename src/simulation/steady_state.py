"""Steady-state detection for open-system simulations."""

from typing import List, Optional

import numpy as np


class SteadyStateDetector:
    """
    Detect thermal steady state by monitoring heat flux stability.

    Steady state is reached when the coefficient of variation (CV) of the
    time-averaged flux drops below a threshold over a sliding window:

        CV = std(J) / |mean(J)| < threshold

    Once triggered, steady state latches — it does not "un-detect".
    """

    def __init__(self, window_size: int = 100, threshold: float = 0.01):
        """
        Args:
            window_size: Number of flux samples in the sliding window.
            threshold: CV threshold below which steady state is declared.
        """
        self.window_size = window_size
        self.threshold = threshold
        self._flux_samples: List[float] = []
        self._reached = False
        self._reached_time: Optional[float] = None

    def add_sample(self, flux: float, time: float) -> bool:
        """
        Append a flux measurement and check for steady state.

        Args:
            flux: Time-averaged flux over the latest measurement interval.
            time: Current simulation time.

        Returns:
            True if steady state has been reached (now or previously).
        """
        self._flux_samples.append(flux)

        if self._reached:
            return True

        if len(self._flux_samples) < self.window_size:
            return False

        recent = self._flux_samples[-self.window_size:]
        mean = np.mean(recent)

        # If mean flux is essentially zero, CV is undefined — not steady
        if abs(mean) < 1e-15:
            return False

        cv = float(np.std(recent) / abs(mean))
        if cv < self.threshold:
            self._reached = True
            self._reached_time = time
            return True

        return False

    @property
    def is_steady_state(self) -> bool:
        """Whether steady state has been detected."""
        return self._reached

    @property
    def steady_state_time(self) -> Optional[float]:
        """Simulation time at which steady state was first detected, or None."""
        return self._reached_time

    def coefficient_of_variation(self) -> Optional[float]:
        """
        Current CV of the flux window, or None if insufficient samples.

        Returns:
            CV value, or None if fewer than window_size samples collected
            or if |mean| ≈ 0.
        """
        if len(self._flux_samples) < self.window_size:
            return None

        recent = self._flux_samples[-self.window_size:]
        mean = np.mean(recent)

        if abs(mean) < 1e-15:
            return None

        return float(np.std(recent) / abs(mean))
