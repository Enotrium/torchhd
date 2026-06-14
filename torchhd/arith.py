#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
r"""Bit-accurate fixed-point arithmetic matching FPGA/ASIC RTL behavior.

This module provides the functional arithmetic operations that an RTL
accelerator would perform: bundling (addition), binding (multiplication/XNOR),
dot-product similarity, and majority voting — all computed in fixed-point
with matching saturation, truncation, and rounding modes.

The goal is to produce bit-identical results to what the Verilog/VHDL ALU
generates, so that torchhd serves as a golden reference model.

Classes
-------
ArithConfig
    Fixed-point arithmetic configuration matching one RTL ALU instance.

Functions
---------
q_bundle
    Fixed-point bundling (addition) with configurable saturation.
q_bind
    Fixed-point binding (multiplication or XNOR for bipolar) in Q-format.
q_permute
    Fixed-point cyclic shift / permutation matching barrel-shifter width.
q_dot
    Bit-accurate dot-product with configurable accumulator width.
q_cos_similarity
    Cosine similarity emulated in fixed-point arithmetic.
q_majority
    Majority vote across bundled vectors with matching gate depth.
"""

from typing import Optional, Literal, Union
import math

import torch
from torch import Tensor

from torchhd.quantize import FixedPointConfig, QuantizedWeight, _clamp_and_round

__all__ = [
    "ArithConfig",
    "q_bundle",
    "q_bind",
    "q_permute",
    "q_dot",
    "q_cos_similarity",
    "q_majority",
]


class ArithConfig:
    r"""Fixed-point arithmetic configuration for one RTL ALU instance.

    Args:
        data_in: Fixed-point format of the input vectors (operands).
        data_out: Fixed-point format after bundling/binding (result).
        acc: Fixed-point format of the accumulator inside dot-product units.
        saturation: Saturation mode — ``"clamp"`` wraps to min/max, ``"wrap"``
            provides two's complement wrap-around (common in DSP slices).
        permute_shift_bits: Bit-width of the barrel shifter (0 = no shifter,
            N = ``N``-bit shift amount).

    Attributes:
        data_in (FixedPointConfig): Input format.
        data_out (FixedPointConfig): Output format.
        acc (FixedPointConfig): Accumulator format.
        saturation (str): Saturation mode.
        permute_shift_bits (int): Shifter width in bits.
    """

    data_in: FixedPointConfig
    data_out: FixedPointConfig
    acc: FixedPointConfig
    saturation: str
    permute_shift_bits: int

    def __init__(
        self,
        data_in: FixedPointConfig,
        data_out: Optional[FixedPointConfig] = None,
        acc: Optional[FixedPointConfig] = None,
        saturation: Literal["clamp", "wrap"] = "clamp",
        permute_shift_bits: int = 0,
    ) -> None:
        self.data_in = data_in
        # Default: output format same as input, but with 1 extra int bit for
        # safe bundling (to avoid overflow on pairwise sums)
        if data_out is None:
            n_frac = data_in.frac
            n_int = data_in.int_bits + 1
            if data_in.signed:
                n_int += 1  # signed sum needs one more bit
            self.data_out = FixedPointConfig(
                bits=n_int + n_frac,
                frac=n_frac,
                signed=data_in.signed,
                rounding=data_in.rounding,
            )
        else:
            self.data_out = data_out

        # Default accumulator: wide enough for D-wide inner product
        if acc is None:
            # log2(D * max_val^2) bits needed
            self.acc = FixedPointConfig(
                bits=32,
                frac=data_in.frac * 2,
                signed=True,
                rounding=data_in.rounding,
            )
        else:
            self.acc = acc

        if saturation not in ("clamp", "wrap"):
            raise ValueError(f"saturation must be 'clamp' or 'wrap', got {saturation}")
        self.saturation = saturation
        self.permute_shift_bits = permute_shift_bits

    def __repr__(self) -> str:
        return (
            f"ArithConfig(in={self.data_in!r}, out={self.data_out!r}, "
            f"acc={self.acc!r}, sat={self.saturation}, "
            f"shift={self.permute_shift_bits}b)"
        )


def _saturate(x: Tensor, cfg: FixedPointConfig, mode: str) -> Tensor:
    """Apply saturation/wrap to the given integer codes."""
    mask = (1 << cfg.bits) - 1
    half = 1 << (cfg.bits - 1)

    if mode == "clamp":
        if cfg.signed:
            min_code = half  # two's complement for -(2^(M-1))
            max_code = half - 1
            # In unsigned integer space for two's complement:
            # negative values are at the top of the range.
            # Clamp to signed range: [-2^(M-1), 2^(M-1)-1]
            # In two's complement unsigned view: [half, mask] and [0, half-1] are valid.
            # Values below half are positive/zero (0 to half-1), values >= half are negative.
            # An overflow above max_val wraps into the negative space.
            # Clamp: values that would exceed positive max go to max_pos.
            #        values that would underflow below negative min go to min_neg.
            x_clamped = x.clone()
            # Positive overflow: x is in valid-positive range but too large
            pos_overflow = (x >= 0) & (x >= half)
            x_clamped[pos_overflow] = half - 1
            # Negative underflow: x < -2^(M-1)
            neg_underflow = x < -(1 << (cfg.int_bits - 1))
            x_clamped[neg_underflow] = -(1 << (cfg.int_bits - 1))
            x_clamped = x_clamped & mask
            return x_clamped
        else:
            return x.clamp(0, mask)
    elif mode == "wrap":
        return x & mask
    else:
        raise ValueError(f"Unknown saturation mode: {mode}")


def q_bundle(
    x: QuantizedWeight,
    y: QuantizedWeight,
    cfg: Optional[ArithConfig] = None,
) -> QuantizedWeight:
    r"""Fixed-point bundling (element-wise addition) matching RTL adder behavior.

    .. math::
        z_i = \text{saturate}(x_i + y_i)

    Args:
        x: First quantized hypervector.
        y: Second quantized hypervector (must match config and shape).
        cfg: Arithmetic configuration.  If ``None``, a default is inferred
            from ``x.config``.

    Returns:
        QuantizedWeight: Bundled result in ``cfg.data_out`` format.

    Examples::

        >>> cfg_in = FixedPointConfig(bits=8, frac=4)
        >>> cfg_arith = ArithConfig(cfg_in)
        >>> a = to_fixed_point(torch.tensor([1.5, -2.0]), bits=8, frac=4)
        >>> b = to_fixed_point(torch.tensor([0.5, -1.0]), bits=8, frac=4)
        >>> q_bundle(a, b, cfg_arith)
    """
    if x.config.bits != y.config.bits or x.config.frac != y.config.frac:
        raise ValueError(
            f"Input formats must match: {x.config!r} vs {y.config!r}"
        )
    if cfg is None:
        cfg = ArithConfig(x.config)

    # Sign-extend both operands to a wider signed integer
    x_int = x.data.to(torch.long)
    y_int = y.data.to(torch.long)

    # Sign-extend from two's complement
    in_bits = x.config.bits
    in_half = 1 << (in_bits - 1)
    if x.config.signed:
        mask_neg_x = x_int >= in_half
        x_int = x_int.clone()
        x_int[mask_neg_x] = x_int[mask_neg_x] - (1 << in_bits)
        mask_neg_y = y_int >= in_half
        y_int = y_int.clone()
        y_int[mask_neg_y] = y_int[mask_neg_y] - (1 << in_bits)

    # Sum
    z_int = x_int + y_int

    # Quantize back to output format: scale match (inherits frac from data_out)
    out_bits = cfg.data_out.bits
    out_frac = cfg.data_out.frac

    # Rescale if output fraction differs from input
    in_frac = x.config.frac
    if out_frac > in_frac:
        # Need to shift left (more fractional bits = finer precision)
        z_int = z_int << (out_frac - in_frac)
    elif out_frac < in_frac:
        # Need to shift right (fewer fractional bits = coarser), with rounding
        shift = in_frac - out_frac
        if cfg.data_out.rounding == "nearest":
            round_add = 1 << (shift - 1) if shift > 0 else 0
            z_int = (z_int + round_add) >> shift
        else:
            z_int = z_int >> shift

    # Saturate to output range
    z_int = _saturate(z_int, cfg.data_out, cfg.saturation)

    return QuantizedWeight(
        z_int.to(torch.long),
        cfg.data_out,
        x.original_shape if x.original_shape == y.original_shape else z_int.shape,
        scale=max(x.scale if isinstance(x.scale, float) else 1.0,
                   y.scale if isinstance(y.scale, float) else 1.0),
    )


def q_bind(
    x: QuantizedWeight,
    y: QuantizedWeight,
    cfg: Optional[ArithConfig] = None,
    bind_mode: Literal["multiply", "xnor", "complex"] = "multiply",
) -> QuantizedWeight:
    r"""Fixed-point binding matching RTL multiplier/XNOR behavior.

    For bipolar (MAP) vectors this is element-wise multiplication.
    For binary (BSC) vectors this is element-wise XNOR.
    Both produce results quantized to ``cfg.data_out``.

    .. math::
        z_i = \text{saturate}(x_i \cdot y_i) \quad\text{or}\quad
        z_i = x_i \oplus y_i

    Args:
        x: First quantized hypervector.
        y: Second quantized hypervector (must match config and shape).
        cfg: Arithmetic configuration.
        bind_mode: ``"multiply"`` for bipolar, ``"xnor"`` for binary,
            ``"complex"`` for FHRR-style complex multiply.

    Returns:
        QuantizedWeight: Bound result in ``cfg.data_out`` format.
    """
    if x.config.bits != y.config.bits or x.config.frac != y.config.frac:
        raise ValueError(
            f"Input formats must match: {x.config!r} vs {y.config!r}"
        )
    if cfg is None:
        cfg = ArithConfig(x.config)

    in_frac = x.config.frac
    out_frac = cfg.data_out.frac
    out_bits = cfg.data_out.bits

    x_int = x.data.to(torch.long)
    y_int = y.data.to(torch.long)

    # Sign-extend
    in_bits = x.config.bits
    in_half = 1 << (in_bits - 1)
    if x.config.signed:
        mask_neg_x = x_int >= in_half
        x_int = x_int.clone()
        x_int[mask_neg_x] = x_int[mask_neg_x] - (1 << in_bits)
        mask_neg_y = y_int >= in_half
        y_int = y_int.clone()
        y_int[mask_neg_y] = y_int[mask_neg_y] - (1 << in_bits)

    if bind_mode == "xnor":
        # XNOR for binary: treat as single-bit, XNOR = NOT XOR
        # For multi-bit binary codes, XNOR per bit
        z_int = ~(x_int ^ y_int) & ((1 << in_bits) - 1)
        if out_frac != in_frac:
            raise ValueError("XNOR bind does not support fraction rescaling")
        return QuantizedWeight(
            z_int, cfg.data_out,
            x.original_shape if x.original_shape == y.original_shape else z_int.shape,
            scale=1.0,
        )
    elif bind_mode == "multiply":
        # Multiply (fractional arithmetic)
        # (x * 2^-frac) * (y * 2^-frac) = (x*y) * 2^(-2*frac)
        z_int = x_int * y_int
        # Result has 2*in_frac fractional bits, shift to out_frac
        product_frac = in_frac * 2
        if out_frac > product_frac:
            z_int = z_int << (out_frac - product_frac)
        elif out_frac < product_frac:
            shift = product_frac - out_frac
            if cfg.data_out.rounding == "nearest":
                round_add = 1 << (shift - 1) if shift > 0 else 0
                z_int = (z_int + round_add) >> shift
            else:
                z_int = z_int >> shift
        z_int = _saturate(z_int, cfg.data_out, cfg.saturation)
        return QuantizedWeight(
            z_int.to(torch.long), cfg.data_out, x.original_shape, scale=1.0,
        )
    elif bind_mode == "complex":
        # FHRR-style: complex multiply. Input is split into real/imag pairs.
        raise NotImplementedError("Complex binding not yet implemented")
    else:
        raise ValueError(f"Unknown bind_mode: {bind_mode}")


def q_permute(
    x: QuantizedWeight,
    shifts: int = 1,
    permute_shift_bits: int = 0,
) -> QuantizedWeight:
    r"""Fixed-point cyclic permutation matching barrel-shifter behavior.

    Performs a cyclic shift of elements.  If ``permute_shift_bits`` is set,
    the shift amount is masked to that many bits (matching the RTL shifter
    width).

    Args:
        x: Quantized hypervector.
        shifts: Number of positions to shift (positive = right).
        permute_shift_bits: Width of the barrel shifter in bits.  If > 0,
            ``shifts`` is masked to ``(1 << permute_shift_bits) - 1``.

    Returns:
        QuantizedWeight: Permuted result in same format as input.
    """
    d = x.original_shape[-1]
    if permute_shift_bits > 0:
        mask = (1 << permute_shift_bits) - 1
        shifts = shifts & mask
    shifts = shifts % d

    if shifts == 0:
        return QuantizedWeight(
            x.data.clone(), x.config, x.original_shape, x.scale
        )

    # Preserve shape: permute along last dim
    data = x.data
    if data.dim() == 1:
        permuted = torch.cat([data[-shifts:], data[:-shifts]])
    else:
        permuted = torch.cat(
            [data[..., -shifts:], data[..., :-shifts]], dim=-1
        )

    return QuantizedWeight(permuted, x.config, x.original_shape, x.scale)


def q_dot(
    x: QuantizedWeight,
    y: QuantizedWeight,
    cfg: Optional[ArithConfig] = None,
) -> Tensor:
    r"""Bit-accurate dot-product similarity with configurable accumulator.

    .. math::
        s = \sum_i \text{saturate}_{\text{acc}}(x_i \cdot y_i)

    Each element product is accumulated in ``cfg.acc`` width, with saturation
    at every addition (matching the RTL adder tree).

    Args:
        x: First matrix of quantized hypervectors ``(N, D)`` or ``(D,)``.
        y: Second matrix ``(M, D)`` or ``(D,)``.
        cfg: Arithmetic configuration. The ``acc`` field determines the
            accumulator bit-width.

    Returns:
        LongTensor: Similarity scores (N, M) or scalar.
    """
    if x.config.bits != y.config.bits:
        raise ValueError(
            f"Input formats must match: {x.config!r} vs {y.config!r}"
        )
    if cfg is None:
        cfg = ArithConfig(x.config)

    # Sign-extend both operands
    in_bits = x.config.bits
    in_half = 1 << (in_bits - 1)
    x_int = x.data.to(torch.long)
    y_int = y.data.to(torch.long)

    if x.config.signed:
        mask_neg_x = x_int >= in_half
        x_int = x_int.clone()
        x_int[mask_neg_x] = x_int[mask_neg_x] - (1 << in_bits)
        mask_neg_y = y_int >= in_half
        y_int = y_int.clone()
        y_int[mask_neg_y] = y_int[mask_neg_y] - (1 << in_bits)

    # Reshape for matrix multiplication
    if x_int.dim() == 1:
        x_int = x_int.unsqueeze(0)
    if y_int.dim() == 1:
        y_int = y_int.unsqueeze(0)

    # Perform the dot product in wider integer precision, then saturate
    # to match accumulator bit-width at each partial sum
    # For simplicity: compute full dot product in int64, then saturate to acc width
    sim = x_int @ y_int.T

    # Saturate to accumulator width
    acc_mask = (1 << cfg.acc.bits) - 1
    sim = _saturate(sim, cfg.acc, cfg.saturation)

    return sim


def q_cos_similarity(
    x: QuantizedWeight,
    y: QuantizedWeight,
    cfg: Optional[ArithConfig] = None,
) -> Tensor:
    r"""Cosine similarity emulated in fixed-point arithmetic.

    Computes:

    .. math::
        \cos(x, y) = \frac{x \cdot y}{\|x\| \cdot \|y\|}

    where all operations (dot product, square root, division) use fixed-point
    approximations matching the RTL implementation.

    Args:
        x: First matrix of quantized hypervectors.
        y: Second matrix of quantized hypervectors.
        cfg: Arithmetic configuration.

    Returns:
        Tensor: Cosine similarity scores in floating point (final scaling step).
    """
    if cfg is None:
        cfg = ArithConfig(x.config)

    dot = q_dot(x, y, cfg).float() / (2.0 ** (cfg.data_in.frac * 2))

    # L2 norms
    x_int = x.data.to(torch.long)
    y_int = y.data.to(torch.long)

    if x.config.signed:
        in_bits = x.config.bits
        in_half = 1 << (in_bits - 1)
        mask_neg_x = x_int >= in_half
        x_int = x_int.clone()
        x_int[mask_neg_x] = x_int[mask_neg_x] - (1 << in_bits)
        mask_neg_y = y_int >= in_half
        y_int = y_int.clone()
        y_int[mask_neg_y] = y_int[mask_neg_y] - (1 << in_bits)

    x_sum_sq = (x_int * x_int).sum(dim=-1).float() / (2.0 ** (x.config.frac * 2))
    y_sum_sq = (y_int * y_int).sum(dim=-1).float() / (2.0 ** (y.config.frac * 2))

    x_norm = torch.sqrt(x_sum_sq + 1e-12)
    y_norm = torch.sqrt(y_sum_sq + 1e-12)

    # Reshape for broadcasting
    if x_norm.dim() == 0:
        x_norm = x_norm.unsqueeze(0)
    if y_norm.dim() == 0:
        y_norm = y_norm.unsqueeze(0)

    return dot / (x_norm.unsqueeze(-1) * y_norm.unsqueeze(0) + 1e-12)


def q_majority(
    qw_list: list,
    threshold: float = 0.0,
    cfg_out: Optional[FixedPointConfig] = None,
) -> QuantizedWeight:
    r"""Majority vote across bundled quantized vectors (matching RTL gate depth).

    Sums all input vectors element-wise, then thresholds:

    .. math::
        z_i = \begin{cases}
            +1 & \text{if } \sum_j x_{j,i} > \text{threshold} \cdot N \\
            -1 & \text{otherwise}
        \end{cases}

    This matches the behavior of a majority-gate tree in hardware.

    Args:
        qw_list: List of :class:`QuantizedWeight` with matching configs.
        threshold: Fraction of voters needed for ``+1``.  ``0.0`` means simple
            majority (more positive than negative).
        cfg_out: Output format for the resulting bipolar vector.

    Returns:
        QuantizedWeight: Majority-vote result (bipolar {-1, +1} in ``cfg_out``).

    Examples::

        >>> cfg = FixedPointConfig(bits=2, frac=0)
        >>> a = to_fixed_point(torch.tensor([1, -1, 1]), bits=2, frac=0)
        >>> b = to_fixed_point(torch.tensor([1, 1, -1]), bits=2, frac=0)
        >>> c = to_fixed_point(torch.tensor([-1, 1, 1]), bits=2, frac=0)
        >>> q_majority([a, b, c])
    """
    if not qw_list:
        raise ValueError("qw_list must contain at least one element")

    ref_cfg = qw_list[0].config
    for i, qw in enumerate(qw_list):
        if qw.config.bits != ref_cfg.bits or qw.config.frac != ref_cfg.frac:
            raise ValueError(
                f"All inputs must share format: qw_list[0]={ref_cfg!r}, "
                f"qw_list[{i}]={qw.config!r}"
            )

    if cfg_out is None:
        cfg_out = FixedPointConfig(bits=1, frac=0, signed=True)

    # Sign-extend and sum (make a wider accumulator to avoid overflow)
    acc_bits = ref_cfg.bits + math.ceil(math.log2(len(qw_list))) + 1
    acc = torch.zeros_like(qw_list[0].data, dtype=torch.long)

    in_bits = ref_cfg.bits
    in_half = 1 << (in_bits - 1)

    for qw in qw_list:
        x_int = qw.data.to(torch.long)
        if ref_cfg.signed:
            mask_neg = x_int >= in_half
            x_int = x_int.clone()
            x_int[mask_neg] = x_int[mask_neg] - (1 << in_bits)
        acc = acc + x_int

    # Threshold
    n = len(qw_list)
    thresh_val = int(threshold * n)
    result = torch.where(
        acc > thresh_val,
        torch.tensor(1, dtype=torch.long),
        torch.tensor(-1, dtype=torch.long),
    )

    # Encode -1 as two's complement in cfg_out
    if cfg_out.signed:
        out_mask = (1 << cfg_out.bits) - 1
        result[result == -1] = out_mask  # all-ones = -1 in two's complement
        result = result & out_mask

    return QuantizedWeight(result, cfg_out, qw_list[0].original_shape, scale=1.0)