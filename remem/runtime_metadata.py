"""Runtime metadata artifacts for reproducible ReMemAgent experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

_RUNTIME_METADATA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Portable description of the Python runtime executing an experiment."""

    schema_version: int
    python_version: str
    python_implementation: str
    operating_system: str
    operating_system_release: str
    machine: str

    @classmethod
    def capture(cls) -> Self:
        """Capture deterministic runtime fields without host-specific identifiers."""
        return cls(
            schema_version=_RUNTIME_METADATA_VERSION,
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            operating_system=platform.system(),
            operating_system_release=platform.release(),
            machine=platform.machine(),
        )

    def verify(self) -> None:
        """Validate the metadata schema and required runtime fields."""
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _RUNTIME_METADATA_VERSION:
            raise ValueError(f"unsupported runtime metadata version: {self.schema_version}")

        for field_name in (
            "python_version",
            "python_implementation",
            "operating_system",
            "operating_system_release",
            "machine",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

    def to_json(self) -> str:
        """Serialize metadata using stable ordering for artifact comparison."""
        self.verify()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize and strictly validate a runtime metadata artifact."""
        if not isinstance(payload, str):
            raise TypeError("runtime metadata payload must be a string")
        try:
            raw_metadata = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("runtime metadata payload must contain valid JSON") from error
        if not isinstance(raw_metadata, dict):
            raise TypeError("runtime metadata JSON root must be an object")

        expected_fields = {
            "schema_version",
            "python_version",
            "python_implementation",
            "operating_system",
            "operating_system_release",
            "machine",
        }
        actual_fields = set(raw_metadata)
        missing_fields = expected_fields - actual_fields
        unknown_fields = actual_fields - expected_fields
        if missing_fields:
            raise ValueError(f"runtime metadata is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(f"runtime metadata contains unknown fields: {sorted(unknown_fields)}")

        metadata = cls(**raw_metadata)
        metadata.verify()
        return metadata

    def save(self, path: str | Path) -> None:
        """Atomically persist verified metadata without leaving partial artifacts."""
        destination = Path(path)
        if not destination.parent.exists():
            raise FileNotFoundError(f"runtime metadata parent does not exist: {destination.parent}")

        payload = self.to_json()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and verify a runtime metadata artifact from disk."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint for runtime compatibility checks."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def matches_current_runtime(self) -> bool:
        """Return whether this artifact describes the currently executing runtime."""
        return self == self.capture()


__all__ = ["RuntimeMetadata"]
