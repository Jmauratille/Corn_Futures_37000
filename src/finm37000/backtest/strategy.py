"""Strategy protocol: map closed-bar history to a target position."""

from typing import Protocol

import pandas as pd

from .models import PortfolioState


class Strategy(Protocol):
    """A callable deciding a target position from closed-bar history."""

    def on_bar(
        self,
        history: pd.DataFrame,
        portfolio: PortfolioState,
    ) -> int:
        """Return the target position after observing the latest closed bar."""
        ...
