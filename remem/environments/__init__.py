"""Environment adapters for external agent benchmarks."""

from remem.environments.base import EnvironmentAdapter, StepResult
from remem.environments.alfworld import AlfWorldAdapter
from remem.environments.webshop import WebShopAdapter

__all__ = ["AlfWorldAdapter", "EnvironmentAdapter", "StepResult", "WebShopAdapter"]
