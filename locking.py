"""Single-process locking for long deterministic experiment runs."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
from typing import Iterator, TextIO


@contextmanager
def exclusive_run_lock(root: Path, purpose: str) -> Iterator[None]:
    """Hold a non-blocking package-wide lock for the lifetime of a runner.

    The lock prevents accidental concurrent trainers from overwriting the same
    deterministic run artifacts.  It is released automatically when the
    process exits, including abnormal termination by the operating system.
    """
    if os.environ.get("QMI_EXPERIMENT_LOCK_HELD") == "1":
        yield
        return

    lock_path = root / "logs" / "experiment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "another process"
        handle.close()
        raise RuntimeError(
            f"Experiment package is already locked by {owner}. "
            "Concurrent training is disabled to protect result integrity."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} purpose={purpose}\n")
    handle.flush()
    try:
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
