"""Regression tests for external benchmark cleanup exception handling."""

import pytest

from experiments.external_benchmark import _close_environment_safely


class InterruptingEnvironment:
    """Environment whose cleanup simulates an external process interrupt."""

    def close(self) -> None:
        """Raise a process-level interrupt instead of returning normally."""

        raise KeyboardInterrupt()


def test_safe_cleanup_does_not_swallow_keyboard_interrupt() -> None:
    """Process interrupts from cleanup remain visible to the caller."""

    with pytest.raises(KeyboardInterrupt):
        _close_environment_safely(InterruptingEnvironment())
