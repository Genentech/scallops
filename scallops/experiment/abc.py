"""Scallop's abstract base class for lazy loading data.

This module defines an abstract base class (ABC) called LazyLoadData, providing a template for
classes that implement lazy loading of values through a data accessor.
"""

from abc import ABC, abstractmethod
from typing import Any


class _LazyLoadData(ABC):
    """Class with data accessor to lazily load values."""

    @property
    @abstractmethod
    def data(self) -> Any:
        pass
