"""Matched classical, Fourier, and exact-statevector quantum recurrent models.

The primary quantum transformation uses an ``RZ-RY-RX`` local rotation block.
The legacy ``RZ-RY-RZ`` block is retained only for an ablation because its
terminal ``RZ`` angle is structurally invisible to computational-basis Pauli-Z
readout when it is the final local operation before measurement/entanglement.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
from torch import Tensor, nn


VALID_ROTATION_SEQUENCES = {"rz_ry_rx", "rz_ry_rz"}


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _ry(theta: Tensor) -> Tensor:
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    out = torch.zeros(theta.shape + (2, 2), dtype=_complex_dtype(theta.dtype), device=theta.device)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def _rx(theta: Tensor) -> Tensor:
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    cdtype = _complex_dtype(theta.dtype)
    out = torch.zeros(theta.shape + (2, 2), dtype=cdtype, device=theta.device)
    out[..., 0, 0] = c
    out[..., 0, 1] = -1j * s.to(cdtype)
    out[..., 1, 0] = -1j * s.to(cdtype)
    out[..., 1, 1] = c
    return out


def _rz(theta: Tensor) -> Tensor:
    cdtype = _complex_dtype(theta.dtype)
    out = torch.zeros(theta.shape + (2, 2), dtype=cdtype, device=theta.device)
    half = theta / 2.0
    out[..., 0, 0] = torch.exp(-1j * half.to(cdtype))
    out[..., 1, 1] = torch.exp(1j * half.to(cdtype))
    return out


def _apply_single_qubit(state: Tensor, gate: Tensor, wire: int, n_qubits: int) -> Tensor:
    """Apply a constant or sample-specific 2x2 gate to a batched statevector."""
    if state.ndim != 2:
        raise ValueError("state must have shape [batch, 2**n]")
    if not 0 <= wire < n_qubits:
        raise ValueError("wire index is outside the register")
    batch = state.shape[0]
    shaped = state.reshape(batch, *([2] * n_qubits))
    axes = [0] + [1 + w for w in range(n_qubits) if w != wire] + [1 + wire]
    inv = np.argsort(axes)
    permuted = shaped.permute(*axes).reshape(batch, -1, 2)
    if gate.ndim == 2:
        updated = torch.einsum("ij,bkj->bki", gate, permuted)
    elif gate.ndim == 3:
        if gate.shape[0] != batch:
            raise ValueError("batched gate and state batch sizes differ")
        updated = torch.einsum("bij,bkj->bki", gate, permuted)
    else:
        raise ValueError("gate must have shape [2,2] or [batch,2,2]")
    restored = updated.reshape(batch, *([2] * (n_qubits - 1)), 2).permute(*inv)
    return restored.reshape(batch, 2**n_qubits)


def _cnot_permutation(n_qubits: int, control: int, target: int, device: torch.device) -> Tensor:
    if control == target:
        raise ValueError("CNOT control and target must differ")
    dim = 2**n_qubits
    perm = np.arange(dim, dtype=np.int64)
    for basis in range(dim):
        control_bit = (basis >> (n_qubits - 1 - control)) & 1
        if control_bit:
            perm[basis] = basis ^ (1 << (n_qubits - 1 - target))
    return torch.tensor(perm, dtype=torch.long, device=device)


def _basis_bits(indices: Tensor, n_qubits: int) -> Tensor:
    """Return computational-basis bits with shape ``[..., n_qubits]``."""
    bits = []
    for wire in range(n_qubits):
        bits.append((indices >> (n_qubits - 1 - wire)) & 1)
    return torch.stack(bits, dim=-1)


def exact_quantum_expectations(
    inputs: Tensor,
    weights: Tensor,
    entangle: bool = True,
    shots: Optional[int] = None,
    readout_error: float = 0.0,
    generator: Optional[torch.Generator] = None,
    rotation_sequence: str = "rz_ry_rx",
) -> Tensor:
    """Simulate an angle-embedded variational circuit and measure Pauli-Z.

    Parameters
    ----------
    inputs:
        Tensor with shape ``[batch, n_qubits]`` containing RY embedding angles.
    weights:
        Tensor with shape ``[layers, n_qubits, 3]``.
    entangle:
        Apply a directed nearest-neighbor CNOT ring after every variational
        layer.
    shots:
        Optional finite-shot joint computational-basis sampling. Training and
        the primary results use exact expectations (``shots=None``).
    readout_error:
        Independent symmetric measurement bit-flip probability. For exact
        expectations this has the analytic attenuation factor ``1-2p``; for
        finite shots it is applied to sampled bits.
    generator:
        Optional seeded ``torch.Generator`` for reproducible measurement draws.
    rotation_sequence:
        ``"rz_ry_rx"`` for the primary all-active circuit or ``"rz_ry_rz"``
        for the legacy ablation.
    """
    if inputs.ndim != 2 or weights.ndim != 3:
        raise ValueError("inputs must be [batch,n_qubits] and weights [layers,n_qubits,3]")
    if rotation_sequence not in VALID_ROTATION_SEQUENCES:
        raise ValueError(
            f"rotation_sequence must be one of {sorted(VALID_ROTATION_SEQUENCES)}, "
            f"got {rotation_sequence!r}"
        )
    if not 0.0 <= readout_error < 0.5:
        raise ValueError("readout_error must be in [0, 0.5)")
    if shots is not None and shots <= 0:
        raise ValueError("shots must be positive")

    batch, n_qubits = inputs.shape
    if n_qubits < 1:
        raise ValueError("at least one qubit is required")
    if weights.shape[1] != n_qubits or weights.shape[2] != 3:
        raise ValueError("quantum weight shape does not match n_qubits")

    cdtype = _complex_dtype(inputs.dtype)
    state = torch.zeros((batch, 2**n_qubits), dtype=cdtype, device=inputs.device)
    state[:, 0] = 1.0 + 0.0j

    # Angle embedding.
    for wire in range(n_qubits):
        state = _apply_single_qubit(state, _ry(inputs[:, wire]), wire, n_qubits)

    # Trainable circuit.
    for layer in range(weights.shape[0]):
        for wire in range(n_qubits):
            phi, theta, omega = weights[layer, wire]
            state = _apply_single_qubit(state, _rz(phi), wire, n_qubits)
            state = _apply_single_qubit(state, _ry(theta), wire, n_qubits)
            final_gate = _rx(omega) if rotation_sequence == "rz_ry_rx" else _rz(omega)
            state = _apply_single_qubit(state, final_gate, wire, n_qubits)
        if entangle and n_qubits > 1:
            for control in range(n_qubits):
                target = (control + 1) % n_qubits
                perm = _cnot_permutation(n_qubits, control, target, state.device)
                state = state[:, perm]

    probs = state.real.square() + state.imag.square()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(torch.finfo(inputs.dtype).eps)

    if shots is not None:
        # Draw complete bit strings so inter-qubit measurement correlations are
        # retained. This is an inference-only sensitivity path.
        samples = torch.multinomial(
            probs,
            num_samples=int(shots),
            replacement=True,
            generator=generator,
        )
        bits = _basis_bits(samples, n_qubits).to(inputs.dtype)
        if readout_error:
            flips = torch.rand(
                bits.shape,
                dtype=inputs.dtype,
                device=inputs.device,
                generator=generator,
            ) < readout_error
            bits = torch.remainder(bits + flips.to(inputs.dtype), 2.0)
        return (1.0 - 2.0 * bits).mean(dim=1)

    basis = torch.arange(2**n_qubits, device=inputs.device)
    outputs = []
    for wire in range(n_qubits):
        bits = (basis >> (n_qubits - 1 - wire)) & 1
        signs = (1.0 - 2.0 * bits.to(inputs.dtype)).unsqueeze(0)
        outputs.append(torch.sum(probs * signs, dim=1))
    z = torch.stack(outputs, dim=1)
    if readout_error:
        z = z * (1.0 - 2.0 * readout_error)
    return z


class TransformCore(nn.Module):
    quantum: bool = False

    def forward(
        self,
        x: Tensor,
        *,
        shots: Optional[int] = None,
        readout_error: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        raise NotImplementedError


class ClassicalCore(TransformCore):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        return torch.tanh(self.linear(x))


class FourierCore(TransformCore):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        return torch.sin(self.linear(x))


class QuantumCore(TransformCore):
    quantum = True

    def __init__(
        self,
        width: int,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
    ):
        super().__init__()
        if rotation_sequence not in VALID_ROTATION_SEQUENCES:
            raise ValueError(f"Unsupported rotation sequence: {rotation_sequence}")
        self.width = width
        self.circuit_layers = circuit_layers
        self.entangle = entangle
        self.rotation_sequence = rotation_sequence
        self.weights = nn.Parameter(torch.empty(circuit_layers, width, 3))
        nn.init.uniform_(self.weights, -math.pi, math.pi)

    def forward(
        self,
        x: Tensor,
        *,
        shots: Optional[int] = None,
        readout_error: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        return exact_quantum_expectations(
            x,
            self.weights,
            entangle=self.entangle,
            shots=shots,
            readout_error=readout_error,
            generator=generator,
            rotation_sequence=self.rotation_sequence,
        )


class GateTransform(nn.Module):
    """Input -> angle bottleneck -> matched nonlinear core -> hidden output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kind: str,
        n_qubits: int = 2,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
    ):
        super().__init__()
        self.kind = kind
        self.pre = nn.Linear(input_dim, n_qubits)
        if kind == "classical":
            self.core: TransformCore = ClassicalCore(n_qubits)
        elif kind == "fourier":
            self.core = FourierCore(n_qubits)
        elif kind == "quantum":
            self.core = QuantumCore(
                n_qubits,
                circuit_layers=circuit_layers,
                entangle=entangle,
                rotation_sequence=rotation_sequence,
            )
        else:
            raise KeyError(f"Unknown transform kind: {kind}")
        self.post = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        x: Tensor,
        *,
        shots: Optional[int] = None,
        readout_error: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        angles = math.pi * torch.tanh(self.pre(x))
        core = self.core(
            angles,
            shots=shots,
            readout_error=readout_error,
            generator=generator,
        )
        return self.post(core)


class LiGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transform_kind: str,
        n_qubits: int = 2,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kind=transform_kind,
            n_qubits=n_qubits,
            circuit_layers=circuit_layers,
            entangle=entangle,
            rotation_sequence=rotation_sequence,
        )
        self.update_x = GateTransform(**kwargs)
        self.candidate_x = GateTransform(**kwargs)
        self.update_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        *,
        shots: Optional[int] = None,
        readout_error: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        flat = x.reshape(batch * steps, -1)
        update_inputs = self.update_x(
            flat, shots=shots, readout_error=readout_error, generator=generator
        ).reshape(batch, steps, self.hidden_dim)
        candidate_inputs = self.candidate_x(
            flat, shots=shots, readout_error=readout_error, generator=generator
        ).reshape(batch, steps, self.hidden_dim)
        outputs = []
        for t in range(steps):
            z = torch.sigmoid(update_inputs[:, t] + self.update_h(h) + self.update_bias)
            candidate = torch.tanh(
                candidate_inputs[:, t] + self.candidate_h(h) + self.candidate_bias
            )
            h = (1.0 - z) * h + z * candidate
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class QuantumLSTMCell(nn.Module):
    """Four-transform QLSTM used for resource-efficiency comparison.

    The cell mirrors the controlled design used for Quantum LiGRU and QGRU:
    each input-to-gate map is a ``GateTransform`` with the same angle
    bottleneck and variational core, while hidden-to-gate maps remain linear.
    This isolates the recurrent-gating cost in quantum circuit evaluations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 2,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kind="quantum",
            n_qubits=n_qubits,
            circuit_layers=circuit_layers,
            entangle=entangle,
            rotation_sequence=rotation_sequence,
        )
        self.input_x = GateTransform(**kwargs)
        self.forget_x = GateTransform(**kwargs)
        self.output_x = GateTransform(**kwargs)
        self.candidate_x = GateTransform(**kwargs)
        self.input_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forget_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.forget_bias = nn.Parameter(torch.ones(hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: Tensor, h: Tensor, **noise: object) -> Tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        flat = x.reshape(batch * steps, -1)
        ix = self.input_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        fx = self.forget_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        ox = self.output_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        gx = self.candidate_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        c = torch.zeros_like(h)
        outputs = []
        for t in range(steps):
            i = torch.sigmoid(ix[:, t] + self.input_h(h) + self.input_bias)
            f = torch.sigmoid(fx[:, t] + self.forget_h(h) + self.forget_bias)
            o = torch.sigmoid(ox[:, t] + self.output_h(h) + self.output_bias)
            g = torch.tanh(gx[:, t] + self.candidate_h(h) + self.candidate_bias)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class QuantumGRUCell(nn.Module):
    """Three-transform QGRU used only as a resource/architecture sensitivity check."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 2,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kind="quantum",
            n_qubits=n_qubits,
            circuit_layers=circuit_layers,
            entangle=entangle,
            rotation_sequence=rotation_sequence,
        )
        self.update_x = GateTransform(**kwargs)
        self.reset_x = GateTransform(**kwargs)
        self.candidate_x = GateTransform(**kwargs)
        self.update_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: Tensor, h: Tensor, **noise: object) -> Tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        flat = x.reshape(batch * steps, -1)
        ux = self.update_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        rx = self.reset_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        cx = self.candidate_x(flat, **noise).reshape(batch, steps, self.hidden_dim)
        outputs = []
        for t in range(steps):
            z = torch.sigmoid(ux[:, t] + self.update_h(h) + self.update_bias)
            r = torch.sigmoid(rx[:, t] + self.reset_h(h) + self.reset_bias)
            candidate = torch.tanh(cx[:, t] + self.candidate_h(r * h) + self.candidate_bias)
            h = (1.0 - z) * h + z * candidate
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class CustomRecurrentRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        recurrent_layers: int,
        dropout: float,
        transform_kind: str,
        n_qubits: int = 2,
        circuit_layers: int = 1,
        entangle: bool = True,
        rotation_sequence: str = "rz_ry_rx",
        quantum_gru: bool = False,
        quantum_lstm: bool = False,
    ):
        super().__init__()
        cells = []
        for layer in range(recurrent_layers):
            layer_input = input_dim if layer == 0 else hidden_dim
            if quantum_lstm:
                cell = QuantumLSTMCell(
                    layer_input,
                    hidden_dim,
                    n_qubits=n_qubits,
                    circuit_layers=circuit_layers,
                    entangle=entangle,
                    rotation_sequence=rotation_sequence,
                )
            elif quantum_gru:
                cell = QuantumGRUCell(
                    layer_input,
                    hidden_dim,
                    n_qubits=n_qubits,
                    circuit_layers=circuit_layers,
                    entangle=entangle,
                    rotation_sequence=rotation_sequence,
                )
            else:
                cell = LiGRUCell(
                    layer_input,
                    hidden_dim,
                    transform_kind=transform_kind,
                    n_qubits=n_qubits,
                    circuit_layers=circuit_layers,
                    entangle=entangle,
                    rotation_sequence=rotation_sequence,
                )
            cells.append(cell)
        self.cells = nn.ModuleList(cells)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: Tensor,
        *,
        shots: Optional[int] = None,
        readout_error: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        output = x
        for layer, cell in enumerate(self.cells):
            h = torch.zeros(x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
            output, _ = cell(
                output,
                h,
                shots=shots,
                readout_error=readout_error,
                generator=generator,
            )
            if layer < len(self.cells) - 1:
                output = self.dropout(output)
        final = self.dropout(output[:, -1])
        return self.readout(final).squeeze(-1)


class TorchRecurrentRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, recurrent_layers: int, dropout: float, kind: str):
        super().__init__()
        recurrent_dropout = dropout if recurrent_layers > 1 else 0.0
        if kind == "gru":
            self.recurrent = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        elif kind == "lstm":
            self.recurrent = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            raise KeyError(kind)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        output, _ = self.recurrent(x)
        return self.readout(self.dropout(output[:, -1])).squeeze(-1)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    uses_egarch: bool
    family: str
    transform: Optional[str] = None
    n_qubits: int = 2
    circuit_layers: int = 1
    entangle: bool = True
    rotation_sequence: str = "rz_ry_rx"


PRIMARY_MODEL_SPECS: Dict[str, ModelSpec] = {
    "gru": ModelSpec("gru", False, "torch"),
    "lstm": ModelSpec("lstm", False, "torch"),
    "ligru": ModelSpec("ligru", False, "custom", "classical"),
    "fourier_ligru": ModelSpec("fourier_ligru", False, "custom", "fourier"),
    "quantum_ligru": ModelSpec("quantum_ligru", False, "custom", "quantum"),
    "egarch_ligru": ModelSpec("egarch_ligru", True, "custom", "classical"),
    "egarch_fourier_ligru": ModelSpec("egarch_fourier_ligru", True, "custom", "fourier"),
    "egarch_quantum_ligru": ModelSpec("egarch_quantum_ligru", True, "custom", "quantum"),
}


def configured_primary_model_spec(name: str, quantum_config: Optional[dict] = None) -> ModelSpec:
    """Return a primary specification with the configured quantum circuit.

    Classical specifications are returned unchanged. Quantum specifications
    inherit the qubit count, depth, entanglement flag, and rotation sequence
    from ``config.yaml`` so checkpoint metadata can be validated exactly.
    """
    if name not in PRIMARY_MODEL_SPECS:
        raise KeyError(f"Unknown primary model: {name}")
    spec = PRIMARY_MODEL_SPECS[name]
    if spec.transform != "quantum" and spec.family not in {"quantum_gru", "quantum_lstm"}:
        return spec
    q = quantum_config or {}
    return replace(
        spec,
        n_qubits=int(q.get("n_qubits", spec.n_qubits)),
        circuit_layers=int(q.get("circuit_layers", spec.circuit_layers)),
        entangle=bool(q.get("entangle", spec.entangle)),
        rotation_sequence=str(q.get("rotation_sequence", spec.rotation_sequence)),
    )


def build_model(
    spec: ModelSpec,
    input_dim: int,
    hidden_dim: int,
    recurrent_layers: int,
    dropout: float,
) -> nn.Module:
    if spec.family == "torch":
        return TorchRecurrentRegressor(input_dim, hidden_dim, recurrent_layers, dropout, spec.name)
    if spec.family == "custom":
        return CustomRecurrentRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            recurrent_layers=recurrent_layers,
            dropout=dropout,
            transform_kind=spec.transform or "classical",
            n_qubits=spec.n_qubits,
            circuit_layers=spec.circuit_layers,
            entangle=spec.entangle,
            rotation_sequence=spec.rotation_sequence,
        )
    if spec.family == "quantum_gru":
        return CustomRecurrentRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            recurrent_layers=recurrent_layers,
            dropout=dropout,
            transform_kind="quantum",
            n_qubits=spec.n_qubits,
            circuit_layers=spec.circuit_layers,
            entangle=spec.entangle,
            rotation_sequence=spec.rotation_sequence,
            quantum_gru=True,
        )
    if spec.family == "quantum_lstm":
        return CustomRecurrentRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            recurrent_layers=recurrent_layers,
            dropout=dropout,
            transform_kind="quantum",
            n_qubits=spec.n_qubits,
            circuit_layers=spec.circuit_layers,
            entangle=spec.entangle,
            rotation_sequence=spec.rotation_sequence,
            quantum_lstm=True,
        )
    raise KeyError(spec.family)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quantum_transform_count_per_timestep(spec: ModelSpec, recurrent_layers: int) -> int:
    if spec.family == "quantum_gru":
        return 3 * recurrent_layers
    if spec.family == "quantum_lstm":
        return 4 * recurrent_layers
    if spec.transform == "quantum":
        return 2 * recurrent_layers
    return 0


def quantum_core_parameter_count(spec: ModelSpec, recurrent_layers: int) -> int:
    transforms = quantum_transform_count_per_timestep(spec, recurrent_layers)
    return transforms * spec.circuit_layers * spec.n_qubits * 3


def structurally_active_quantum_parameter_count(spec: ModelSpec, recurrent_layers: int) -> int:
    transforms = quantum_transform_count_per_timestep(spec, recurrent_layers)
    if transforms == 0:
        return 0
    if spec.rotation_sequence == "rz_ry_rx":
        active_per_qubit_layer = 3
    elif spec.rotation_sequence == "rz_ry_rz":
        active_per_qubit_layer = 2
    else:
        raise ValueError(f"Unsupported rotation sequence: {spec.rotation_sequence}")
    return transforms * spec.circuit_layers * spec.n_qubits * active_per_qubit_layer
