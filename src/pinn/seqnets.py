"""Sequence architectures for the pointwise PINN: a Transformer and an RNN.

Both follow PINNsFormer (Zhao, Ding & Prakash, ICLR 2024, arXiv:2307.11833),
which was proposed specifically against PINN failure modes of the kind this
project studies. A single query point (x, y, t) is expanded into a short
*pseudo-sequence* in time,

    (x, y, t), (x, y, t + dt), ..., (x, y, t + (k-1) dt),

each element is embedded, a sequence model mixes them, and a head maps the
result to (u, v, p). The question for this project is whether letting the
network see a point's own temporal neighbourhood is enough to stop the
collapse onto the steady base flow, the way causal time-marching does.

Two encoders share the embedding and the head, so the only difference between
them is how the pseudo-sequence is mixed:

    PINNsFormer  self-attention encoder + cross-attention decoder, the
                 published architecture with its wavelet activation
    PINNsGRU     a GRU run over the same sequence -- the recurrent analogue,
                 so "attention vs recurrence" is the controlled comparison

**Pointwise readout.** PINNsFormer applies the PDE loss at all k pseudo-times.
Here the network returns the prediction of the token for t itself, so the
model is an ordinary function (x, y, t) -> (u, v, p) and drops straight into
`NavierStokes2DUnsteadyPDE`, the evaluation, and the windowed marching without
any change to them. With the published dt = 1e-4 (in the network's normalised
time) the other k-1 tokens sit within 1e-4 of t, so their residuals are
near-copies of this one; dropping them saves a factor of k in derivative cost.

Neither architecture mixes information *across* query points -- attention and
recurrence act only within one point's pseudo-sequence -- which is what keeps
the per-point JVP derivatives of the PDE operator correct.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class WaveAct(nn.Module):
    """PINNsFormer's wavelet activation, w1 sin(x) + w2 cos(x)."""

    @nn.compact
    def __call__(self, x):
        w1 = self.param("w1", nn.initializers.ones, ())
        w2 = self.param("w2", nn.initializers.ones, ())
        return w1 * jnp.sin(x) + w2 * jnp.cos(x)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int = 256

    @nn.compact
    def __call__(self, x):
        x = WaveAct()(nn.Dense(self.d_ff)(x))
        x = WaveAct()(nn.Dense(self.d_ff)(x))
        return nn.Dense(self.d_model)(x)


class EncoderLayer(nn.Module):
    d_model: int
    heads: int

    @nn.compact
    def __call__(self, x):
        h = WaveAct()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(h, h)
        return x + FeedForward(self.d_model)(WaveAct()(x))


class DecoderLayer(nn.Module):
    d_model: int
    heads: int

    @nn.compact
    def __call__(self, x, memory):
        h = WaveAct()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(h, memory)
        return x + FeedForward(self.d_model)(WaveAct()(x))


def pseudo_sequence(x, k, dt):
    """(N, 3) -> (N, k, 3): the query point repeated at t, t + dt, ..."""
    offs = jnp.arange(k, dtype=x.dtype) * dt
    seq = jnp.repeat(x[:, None, :], k, axis=1)
    return seq.at[:, :, 2].add(offs[None, :])


class Head(nn.Module):
    d_hidden: int
    d_out: int

    @nn.compact
    def __call__(self, x):
        x = WaveAct()(nn.Dense(self.d_hidden)(x))
        x = WaveAct()(nn.Dense(self.d_hidden)(x))
        return nn.Dense(self.d_out)(x)


class PINNsFormer(nn.Module):
    """Transformer PINN; defaults are the published ones (d_model 32,
    hidden 512, one encoder and one decoder layer, two heads, k = 5,
    dt = 1e-4)."""

    d_out: int = 3
    d_model: int = 32
    d_hidden: int = 512
    n_layers: int = 1
    heads: int = 2
    k: int = 5
    dt: float = 1e-4

    @nn.compact
    def __call__(self, x):
        src = nn.Dense(self.d_model)(pseudo_sequence(x, self.k, self.dt))
        mem = src
        for _ in range(self.n_layers):
            mem = EncoderLayer(self.d_model, self.heads)(mem)
        mem = WaveAct()(mem)
        out = src
        for _ in range(self.n_layers):
            out = DecoderLayer(self.d_model, self.heads)(out, mem)
        out = WaveAct()(out)
        return Head(self.d_hidden, self.d_out)(out[:, 0])  # token for t itself


class PINNsGRU(nn.Module):
    """The recurrent analogue: same embedding and head, GRU instead of attention.

    The GRU reads the pseudo-sequence backwards in time, ending on the token
    for t, so its final state -- the one read out -- has just consumed t, with
    the rest of the neighbourhood carried in its memory.
    """

    d_out: int = 3
    d_model: int = 32
    d_hidden: int = 512
    k: int = 5
    dt: float = 1e-4

    @nn.compact
    def __call__(self, x):
        src = nn.Dense(self.d_model)(pseudo_sequence(x, self.k, self.dt))
        cell = nn.GRUCell(features=self.d_model)
        h = jnp.zeros((x.shape[0], self.d_model), x.dtype)
        for i in reversed(range(self.k)):  # unrolled: k is 5
            h, _ = cell(h, src[:, i])
        return Head(self.d_hidden, self.d_out)(WaveAct()(h))


def build_seqnet(kind: str, d_model: int, d_hidden: int, k: int, dt: float,
                 heads: int = 2, n_layers: int = 1):
    if kind == "pinnsformer":
        return PINNsFormer(d_model=d_model, d_hidden=d_hidden, n_layers=n_layers,
                           heads=heads, k=k, dt=dt)
    if kind == "pinns_gru":
        return PINNsGRU(d_model=d_model, d_hidden=d_hidden, k=k, dt=dt)
    raise ValueError(kind)


SEQ_NETS: Sequence[str] = ("pinnsformer", "pinns_gru")
