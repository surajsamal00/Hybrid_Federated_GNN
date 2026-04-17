"""
stream.py
---------
StreamQueue: simulates a real-time stream of new customers.

On creation it holds back a portion of the full dataset.
Each call to `pop_batch()` releases a small batch and distributes
it randomly across the 3 banks — mimicking new loan applications
arriving at different branches over time.
"""

import numpy as np
import pandas as pd


class StreamQueue:
    """
    Parameters
    ----------
    df           : DataFrame  – held-out data (already labelled)
    n_banks      : int        – number of banks (default 3)
    batch_size   : int        – customers released per federation round
    random_state : int
    """

    def __init__(self, df, n_banks=3, batch_size=75, random_state=42):
        self.n_banks    = n_banks
        self.batch_size = batch_size
        rng = np.random.RandomState(random_state)

        # shuffle and store
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self._rows = df

        # pointer into _rows
        self._ptr = 0

    # ── API ───────────────────────────────────────────────────────────────────

    def is_empty(self):
        return self._ptr >= len(self._rows)

    def remaining(self):
        return max(0, len(self._rows) - self._ptr)

    def pop_batch(self):
        """
        Returns a list of `n_banks` DataFrames (one per bank).
        Each contains a random slice of the next `batch_size` rows.
        If fewer than batch_size rows remain, returns what's left.
        """
        if self.is_empty():
            return [pd.DataFrame() for _ in range(self.n_banks)]

        end  = min(self._ptr + self.batch_size, len(self._rows))
        batch = self._rows.iloc[self._ptr:end].copy()
        self._ptr = end

        # Randomly assign each row to one of the n_banks
        bank_ids = np.random.randint(0, self.n_banks, size=len(batch))
        splits   = [batch[bank_ids == b].reset_index(drop=True)
                    for b in range(self.n_banks)]
        return splits
