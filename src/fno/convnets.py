"""Convolutional and recurrent forecasters, as drop-in rivals to the FNO.

Phase 5 asks whether a physics term helps an operator forecast the wake beyond
its trained horizon. That answer was obtained for one backbone, the FNO. These
two let the same question be asked of the other families commonly used for
the job, under the identical data, loss, rollout training and evaluation:

    UNet       a CNN: local 3x3 convolutions in an encoder-decoder with skip
               connections. Unlike the FNO it assumes nothing periodic, so
               it needs no padding against the non-periodic wake.
    ConvGRU    an RNN: the PhyCRNet pattern (Ren et al., CMAME 2022) of a
               strided conv encoder, a convolutional GRU cell, and a decoder.
               Its hidden state is carried from one forecast step to the next
               across the whole rollout, so -- unlike the FNO and the U-Net,
               which see only the last two frames -- it has a memory of the
               trajectory so far.

Both take the same input as the FNO, two stacked frames (prev, now) of the
normalised fluctuation (4 channels), and return the next frame (2 channels).
Both down-sample by powers of two, so the grid is zero-padded up to a multiple
of 2^levels inside the model and cropped back on the way out.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def _pad_to(x, mult):
    nx, ny = x.shape[1], x.shape[2]
    px, py = (-nx) % mult, (-ny) % mult
    return jnp.pad(x, ((0, 0), (0, px), (0, py), (0, 0))), (nx, ny)


class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.Conv(self.features, (3, 3))(x))
        return nn.gelu(nn.Conv(self.features, (3, 3))(x))


class UNet(nn.Module):
    """Encoder-decoder CNN with skip connections."""

    features: Sequence[int] = (32, 64, 128, 256)
    out_channels: int = 2

    @nn.compact
    def __call__(self, x):
        x, (nx, ny) = _pad_to(x, 2 ** (len(self.features) - 1))
        skips = []
        for i, f in enumerate(self.features):
            x = ConvBlock(f)(x)
            if i < len(self.features) - 1:
                skips.append(x)
                x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        for f, s in zip(reversed(self.features[:-1]), reversed(skips)):
            x = nn.ConvTranspose(f, (2, 2), strides=(2, 2))(x)
            x = ConvBlock(f)(jnp.concatenate([x, s], axis=-1))
        return nn.Conv(self.out_channels, (1, 1))(x)[:, :nx, :ny]


class ConvGRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, h, x):
        xh = jnp.concatenate([x, h], axis=-1)
        z = nn.sigmoid(nn.Conv(self.features, (3, 3), name="z")(xh))
        r = nn.sigmoid(nn.Conv(self.features, (3, 3), name="r")(xh))
        n = jnp.tanh(nn.Conv(self.features, (3, 3), name="n")(
            jnp.concatenate([x, r * h], axis=-1)))
        return (1.0 - z) * h + z * n


class ConvGRUNet(nn.Module):
    """Strided encoder -> ConvGRU at 1/4 resolution -> decoder with a skip."""

    hidden: int = 64
    base: int = 32
    out_channels: int = 2

    def init_hidden(self, batch, grid):
        nx, ny = -(-grid[0] // 4), -(-grid[1] // 4)  # ceil: the grid is padded to x4
        return jnp.zeros((batch, nx, ny, self.hidden), jnp.float32)

    @nn.compact
    def __call__(self, x, h):
        x, (nx, ny) = _pad_to(x, 4)
        e1 = ConvBlock(self.base)(x)                                    # 1
        e2 = nn.gelu(nn.Conv(2 * self.base, (3, 3), strides=(2, 2))(e1))  # 1/2
        e3 = nn.gelu(nn.Conv(self.hidden, (3, 3), strides=(2, 2))(e2))    # 1/4
        h = ConvGRUCell(self.hidden)(h, e3)
        d = nn.gelu(nn.ConvTranspose(2 * self.base, (2, 2), strides=(2, 2))(h))
        d = ConvBlock(2 * self.base)(jnp.concatenate([d, e2], axis=-1))
        d = nn.gelu(nn.ConvTranspose(self.base, (2, 2), strides=(2, 2))(d))
        d = ConvBlock(self.base)(jnp.concatenate([d, e1], axis=-1))
        return nn.Conv(self.out_channels, (1, 1))(d)[:, :nx, :ny], h


def count_params(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
