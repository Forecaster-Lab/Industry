from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class QueryContext:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class BaseDataProvider(ABC):
    """Base interface for any data source (database/API/csv)."""

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        """Reserved database hook. Keeps compatibility for future DB integration."""
        self.connected = True

    @abstractmethod
    def fetch(self, context: QueryContext) -> pd.DataFrame:
        raise NotImplementedError
