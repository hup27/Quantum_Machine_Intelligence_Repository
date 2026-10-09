"""All-input terminal-layer compiler (column-state, rightmost-first convention).

The acted-on quantum register is measured and discarded. Earlier local layers
and their CNOT networks remain quantum operations. Only the final monomial
network is implemented as a relabeling of complete measured bit strings.
"""
from __future__ import annotations
import copy
from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
from .models import QuantumCore, _rx, _ry, _rz, _apply_single_qubit, _cnot_permutation, _basis_bits, _complex_dtype

PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],dtype=complex)


def local_matrix(weights: Tensor, sequence: str) -> Tensor:
    """Return U for chronological RZ, RY, (RX or RZ)."""
    value = _ry(weights[..., 1]) @ _rz(weights[..., 0])
    if sequence == 'rz_ry_rx': return _rx(weights[..., 2]) @ value
    if sequence == 'rz_ry_rz': return _rz(weights[..., 2]) @ value
    if sequence == 'rz_ry': return value
    raise ValueError(sequence)


def compile_angles(weights: Tensor, sequence: str) -> tuple[Tensor, Tensor]:
    """Return chronological [phi,theta] and a per-site Frobenius certificate.

    C = Ry(theta) Rz(phi), C^dagger Z C = U^dagger Z U. The Frobenius
    residual is an input-independent certificate, not a sampled-state error.
    """
    u=local_matrix(weights,sequence)
    z=torch.tensor([[1.,0.],[0.,-1.]],dtype=u.dtype,device=u.device)
    axis=u.conj().transpose(-2,-1) @ z @ u
    nx=axis[...,0,1].real
    ny=-axis[...,0,1].imag
    nz=axis[...,0,0].real
    radius=torch.sqrt(nx.square()+ny.square())
    theta=torch.atan2(radius,nz)
    phi=torch.where(radius <= 4*torch.finfo(weights.dtype).eps,
                    torch.zeros_like(radius),torch.atan2(ny,-nx))
    angles=torch.stack((phi,theta),dim=-1)
    c=local_matrix(angles,'rz_ry')
    delta=c.conj().transpose(-2,-1) @ z @ c-axis
    return angles,torch.linalg.matrix_norm(delta,ord='fro',dim=(-2,-1))


def ring_labels(n: int, entangle: bool) -> Tensor:
    """Map pre-ring computational-basis labels c to reported labels b."""
    labels=torch.arange(2**n,dtype=torch.long)
    if entangle and n>1:
        for control in range(n):
            target=(control+1)%n
            bit=(labels >> (n-1-control))&1
            labels=labels ^ (bit << (n-1-target))
    return labels


class CompiledQuantumCore(nn.Module):
    quantum=True
    def __init__(self, original: QuantumCore):
        super().__init__()
        if original.rotation_sequence == 'rz_ry':
            raise ValueError('Two-angle training cores are already reduced; not in compilation audit.')
        self.width=original.width
        self.circuit_layers=original.circuit_layers
        self.entangle=original.entangle
        self.rotation_sequence=original.rotation_sequence
        self.prefix_weights=nn.Parameter(original.weights[:-1].detach().clone())
        angles, certificate=compile_angles(original.weights[-1].detach(),self.rotation_sequence)
        self.terminal_angles=nn.Parameter(angles.clone())
        self.register_buffer('output_labels',ring_labels(self.width,self.entangle).to(original.weights.device))
        # Certificate is metadata, not a circuit or model parameter.
        self.certificate_frobenius=certificate.detach().cpu().tolist()

    def probabilities(self, x: Tensor) -> Tensor:
        """Exact pre-relabeling probabilities for arbitrary preceding circuit layers."""
        batch,n=x.shape
        state=torch.zeros((batch,2**n),dtype=_complex_dtype(x.dtype),device=x.device)
        state[:,0]=1
        for wire in range(n): state=_apply_single_qubit(state,_ry(x[:,wire]),wire,n)
        for weights in self.prefix_weights:
            for wire in range(n):
                # Preserve the earlier chronological elementary-gate sequence.
                state=_apply_single_qubit(state,_rz(weights[wire,0]),wire,n)
                state=_apply_single_qubit(state,_ry(weights[wire,1]),wire,n)
                gate=_rx(weights[wire,2]) if self.rotation_sequence=='rz_ry_rx' else _rz(weights[wire,2])
                state=_apply_single_qubit(state,gate,wire,n)
            if self.entangle and n>1:
                for control in range(n):
                    state=state[:,_cnot_permutation(n,control,(control+1)%n,state.device)]
        for wire in range(n):
            state=_apply_single_qubit(state,_rz(self.terminal_angles[wire,0]),wire,n)
            state=_apply_single_qubit(state,_ry(self.terminal_angles[wire,1]),wire,n)
        probs=state.real.square()+state.imag.square()
        return probs/probs.sum(dim=1,keepdim=True).clamp_min(torch.finfo(x.dtype).eps)

    def forward(self,x:Tensor,*,shots:Optional[int]=None,readout_error:float=0.,
                generator:Optional[torch.Generator]=None)->Tensor:
        if not 0 <= readout_error < .5: raise ValueError('readout_error must be in [0,0.5).')
        if shots is not None and shots<=0: raise ValueError('shots must be positive.')
        probs=self.probabilities(x)
        if shots is not None:
            c=torch.multinomial(probs,int(shots),replacement=True,generator=generator)
            b=self.output_labels[c]
            bits=_basis_bits(b,self.width).to(x.dtype)
            if readout_error:
                flips=torch.rand(bits.shape,dtype=x.dtype,device=x.device,generator=generator)<readout_error
                bits=torch.remainder(bits+flips.to(x.dtype),2.)
            return (1-2*bits).mean(dim=1)
        signs=1-2*_basis_bits(self.output_labels,self.width).to(x.dtype)
        return (probs @ signs)*(1-2*readout_error)


def compile_model(model:nn.Module)->tuple[nn.Module,list[dict]]:
    """Copy a fitted model and replace each eligible terminal core."""
    result=copy.deepcopy(model);certificates=[]
    names=[name for name,module in result.named_modules() if isinstance(module,QuantumCore)]
    for name in names:
        old=result.get_submodule(name);new=CompiledQuantumCore(old)
        parent,_,child=name.rpartition('.')
        setattr(result.get_submodule(parent),child,new)
        certificates.append({'core':name,'residual_frobenius':new.certificate_frobenius,
                             'qubits':new.width,'retained_prefix_layers':new.circuit_layers-1})
    if not names: raise ValueError('No quantum cores found.')
    return result,certificates
