"""Predeclared architecture-ablation model specifications."""
from __future__ import annotations

from .models import ModelSpec


def architecture_ablation_specs() -> list[ModelSpec]:
    """Return the exact COVID-fold architecture sensitivity set."""
    return [
        ModelSpec("quantum_ligru_no_entanglement", False, "custom", "quantum", 2, 1, False),
        ModelSpec("quantum_ligru_depth2", False, "custom", "quantum", 2, 2, True),
        ModelSpec("quantum_ligru_4qubit", False, "custom", "quantum", 4, 1, True),
        ModelSpec("quantum_gru", False, "quantum_gru", "quantum", 2, 1, True),
        ModelSpec("egarch_quantum_ligru_no_entanglement", True, "custom", "quantum", 2, 1, False),
        ModelSpec(
            "quantum_ligru_legacy_rzryrz",
            False,
            "custom",
            "quantum",
            2,
            1,
            True,
            "rz_ry_rz",
        ),
        ModelSpec(
            "egarch_quantum_ligru_legacy_rzryrz",
            True,
            "custom",
            "quantum",
            2,
            1,
            True,
            "rz_ry_rz",
        ),
    ]
