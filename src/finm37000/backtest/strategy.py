from typing import Protocol

import pandas as pd

from .models import PortfolioState


class Strategy(Protocol):
    def on_bar(
        self,
        history: pd.DataFrame,
        portfolio: PortfolioState,
    ) -> int:
        """Return the target position after observing the latest closed bar."""
        ...