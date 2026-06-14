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
r"""Fixed-point quantization and hardware export for hyperdimensional computing.

This module bridges the gap between floating-point hypervectors as used during
training and the fixed-point / integer representations required for FPGA and
ASIC synthesis.  It provides:

* **Quantization functions** — convert tensors to Q-format fixed-point integers.
* **Packing utilities** — pack quantized vectors into dense bit-arrays ready for
  hardware memory (e.g. BRAM, ROM).
* **Export helpers** — render weights as synthesizable Verilog ``localparam``
  declarations, C header arrays, or raw binary files.

Typical Quantization Pipeline
-----------------------------
1. Train a model using standard torchhd floating-point tensors.
2. Quantize the hypervectors with :func:`to_fixed_point` or :func:`quantize_bipolar`:
   ``qv = quantize.to_fixed_point(weight, bits=4, frac=2)``.
3. Pack and export:
   ``packed = quantize.pack_bits(qv, bits_per_element=4, order="big")``.
4. Write to file:
   ``quantize.export_verilog(packed, "hd_weights.sv")``.

Supported Formats
-----------------
* Signed fixed-point ``Q<M.N>`` where *M* integer bits and *N* fractional bits.
* Bipolar (ternary) quantization to {−1, 0, +1}.
* Block-floating point (shared exponent per row / per tensor).
* Packed bit-arrays for dense binary storage.

Exported Output Formats
-----------------------
* Verilog ``localparam`` arrays for direct FPGA synthesis.
* C ``const`` arrays for embedded firmware.
* Raw binary (``.bin``) for memory images.
"""

from typing import Optional, Tuple, Union, Literal
import struct

import torch
from torch import Tensor


__all__ = [
    "to_fixed_point",
    "from_fixed_point",
    "quantize_bipolar",
    "block_float_quantize",
    "pack_bits",
    "unpack_bits",
    "export_verilog",
    "export_c_header",
    "export_binary",
    "FixedPointConfig",
    "QuantizedWeight",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FixedPointConfig:
    r"""Fixed-point format descriptor.

    Args:
        bits: Total bit-width including sign.
        frac: Number of fractional bits (Q<M.N> notation: M = bits - frac).
        signed: Whether the format is signed (two's complement). Default: ``True``.
        rounding: Rounding mode — ``"floor"``, ``"nearest"``, or ``"round"``
            (ties to even).  Default: ``"nearest"``.

    Attributes:
        bits (int): Total bit-width.
        frac (int): Fractional bits.
        int_bits (int): Integer bits (computed as ``bits - frac``).
        min_val (float): Minimum representable value.
        max_val (float): Maximum representable value.
        step (float): Smallest representable difference (one LSB).
    """

    bits: int
    frac: int
    signed: bool
    rounding: str

    def __init__(
        self,
        bits: int,
        frac: int,
        *,
        signed: bool = True,
        rounding: str = "nearest",
    ) -> None:
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        if frac < 0 or frac > bits:
            raise ValueError(f"frac must be in [0, {bits}], got {frac}")

        self.bits = bits
        self.frac = frac
        self.signed = signed
        self.rounding = rounding

    @property
    def int_bits(self) -> int:
        return self.bits - self.frac

    @property
    def step(self) -> float:
        return 2.0 ** (-self.frac)

    @property
    def min_val(self) -> float:
        if self.signed:
            return -(2.0 ** (self.int_bits - 1))
        else:
            return 0.0

    @property
    def max_val(self) -> float:
        if self.signed:
            return (2.0 ** (self.int_bits - 1)) - self.step
        else:
            return (2.0 ** self.int_bits) - self.step

    def __repr__(self) -> str:
        fmt = "Q{}.{}s" if self.signed else "Q{}.{}u"
        return fmt.format(self.int_bits, self.frac)


# ---------------------------------------------------------------------------
# Quantized weight container
# ---------------------------------------------------------------------------

class QuantizedWeight:
    r"""Container holding quantized integer weights together with metadata.

    Attributes:
        data (LongTensor): Quantized integer values (raw hardware representation).
        config (FixedPointConfig): The fixed-point configuration used.
        original_shape (torch.Size): Shape of the unquantized tensor.
        scale (float): Global scale factor, if any was applied prior to quantization.
    """

    data: Tensor
    config: FixedPointConfig
    original_shape: torch.Size
    scale: float

    def __init__(
        self,
        data: Tensor,
        config: FixedPointConfig,
        original_shape: torch.Size,
        scale: float = 1.0,
    ) -> None:
        self.data = data
        self.config = config
        self.original_shape = original_shape
        self.scale = scale

    def __repr__(self) -> str:
        bits = self.config.bits
        return (
            f"QuantizedWeight(shape={tuple(self.data.shape)}, "
            f"dtype={self.data.dtype}, "
            f"format={self.config!r}, "
            f"range={self.data.min().item():.1f}..{self.data.max().item():.1f}, "
            f"scale={self.scale:.6g})"
        )


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------

def _clamp_and_round(
    x: Tensor,
    cfg: FixedPointConfig,
) -> Tensor:
    """Clamp to representable range and quantize to LSB multiples."""
    min_v = cfg.min_val
    max_v = cfg.max_val
    scale = 2.0 ** cfg.frac  # multiply by 2^frac to move to integer domain

    # Clamp
    x_c = x.clamp(min_v, max_v)

    # Scale to integer domain
    x_scaled = x_c * scale

    # Round
    if cfg.rounding == "nearest":
        x_rounded = torch.round(x_scaled)
    elif cfg.rounding == "floor":
        x_rounded = torch.floor(x_scaled)
    elif cfg.rounding == "round":
        # ties-to-even
        x_rounded = torch.round(x_scaled)
    else:
        raise ValueError(f"Unknown rounding mode: {cfg.rounding}")

    # Convert to integer (two's complement for negative values)
    max_code = (1 << cfg.bits) - 1
    half = 1 << (cfg.bits - 1)

    x_int = x_rounded.to(torch.long)

    if cfg.signed:
        # Wrap negatives into two's complement
        x_int = x_int & max_code

    return x_int


def to_fixed_point(
    input: Tensor,
    bits: int,
    frac: int,
    *,
    signed: bool = True,
    rounding: str = "nearest",
    scale: Optional[float] = None,
    per_row_scale: bool = False,
) -> QuantizedWeight:
    r"""Quantize a floating-point tensor to fixed-point.

    Converts each element :math:`v` to an integer code:

    .. math::
        q = \operatorname{clamp}\left(
            \operatorname{round}\left(v \cdot 2^{\text{frac}}\right),
            0,\; 2^{\text{bits}} - 1
        \right)

    For signed formats the result is two's complement (the caller can interpret
    the integer codes as signed by checking ``cfg.signed``).

    Args:
        input (Tensor): Floating-point tensor of any shape.
        bits (int): Total bit-width.
        frac (int): Number of fractional bits.
        signed (bool, optional): Two's complement signed format. Default: ``True``.
        rounding (str, optional): ``"nearest"``, ``"floor"``, or ``"round"``.
            Default: ``"nearest"``.
        scale (float, optional): Pre-scale input by this factor before quantization.
            Useful for normalizing weights to fully use the available range.
        per_row_scale (bool, optional): If ``True``, compute a separate scale for
            each row of a 2-D tensor (block-floating point per row).  The resulting
            ``QuantizedWeight.scale`` will be a 1-D tensor of per-row scales.

    Returns:
        QuantizedWeight: Object containing the integer codes and metadata.

    Examples::

        >>> w = torch.randn(3, 512) * 0.5
        >>> qw = quantize.to_fixed_point(w, bits=8, frac=4)
        >>> qw.data.shape
        torch.Size([3, 512])
        >>> qw.data.dtype
        torch.int64
    """
    cfg = FixedPointConfig(bits=bits, frac=frac, signed=signed, rounding=rounding)

    if per_row_scale and input.dim() == 2:
        # Per-row block-floating point
        abs_max = input.abs().max(dim=1, keepdim=True).values
        eps = 1e-12
        row_scales = (2.0 ** (cfg.int_bits - 1) - cfg.step) / (abs_max + eps)
        scaled_input = input * row_scales
        row_scales = row_scales * (2.0 ** cfg.frac)  # account for frac in scale
        q_int = _clamp_and_round(scaled_input, cfg)
        return QuantizedWeight(q_int, cfg, input.shape, scale=row_scales)
    else:
        if scale is not None:
            scaled_input = input * scale
        else:
            scaled_input = input
        q_int = _clamp_and_round(scaled_input, cfg)
        return QuantizedWeight(q_int, cfg, input.shape, scale=scale or 1.0)


def from_fixed_point(qw: QuantizedWeight) -> Tensor:
    r"""Dequantize a :class:`QuantizedWeight` back to floating point.

    .. math::
        v = \left(\frac{q}{2^{\text{frac}}}\right) / \text{scale}

    where for signed two's complement the integer code *q* is sign-extended.

    Args:
        qw (QuantizedWeight): Previously quantized weight object.

    Returns:
        Tensor: Dequantized floating-point tensor with the original shape.

    Examples::

        >>> w = torch.randn(3, 512)
        >>> qw = quantize.to_fixed_point(w, bits=8, frac=4)
        >>> w_rec = quantize.from_fixed_point(qw)
        >>> (w - w_rec).abs().max() < 0.1
        True
    """
    cfg = qw.config
    data = qw.data.float()

    if cfg.signed:
        max_code = (1 << cfg.bits) - 1
        half = 1 << (cfg.bits - 1)
        # Sign-extend: values >= half are negative
        mask_neg = data >= half
        data = data.clone()
        data[mask_neg] = data[mask_neg] - (1 << cfg.bits)

    # Convert from integer domain back to float
    data = data / (2.0 ** cfg.frac)

    if isinstance(qw.scale, Tensor):
        # If the scale is a per-block or per-row scale (fewer elements than data),
        # reshape both to a block layout, divide, then reshape back.
        if qw.scale.numel() != 1 and data.numel() % qw.scale.numel() == 0:
            block_size = data.numel() // qw.scale.numel()
            flat = data.reshape(-1, block_size)
            scale_r = qw.scale.reshape(-1, 1)
            flat = flat / scale_r
            data = flat.reshape(qw.original_shape)
        elif data.numel() == qw.scale.numel():
            data = data / qw.scale
        else:
            data = data / qw.scale
    else:
        data = data / qw.scale

    return data.reshape(qw.original_shape)


def quantize_bipolar(
    input: Tensor,
    threshold: float = 0.0,
    *,
    ternary: bool = False,
    zero_thresh: Optional[float] = None,
) -> Tensor:
    r"""Quantize to bipolar {−1, +1} or ternary {−1, 0, +1}.

    This is the standard hypervector binarization used for efficient
    similarity computation in hardware.

    Args:
        input (Tensor): Any floating-point tensor.
        threshold (float, optional): Values > ``threshold`` map to +1, otherwise
            to −1 (or 0 in ternary mode).  Default: ``0.0``.
        ternary (bool, optional): If ``True``, produce ternary output with values
            {−1, 0, +1}.  When ``False`` only {−1, +1} are produced.
        zero_thresh (float, optional): In ternary mode, absolute values below
            this threshold are set to 0.  Default: ``None`` uses ``threshold``.

    Returns:
        Tensor: Tensor with the same shape and device as ``input``, dtype ``torch.int8``.

    Examples::

        >>> x = torch.tensor([-0.5, 0.0, 0.3, 0.6])
        >>> quantize_bipolar(x)
        tensor([-1, -1,  1,  1], dtype=torch.int8)
        >>> quantize_bipolar(x, threshold=0.3)
        tensor([-1, -1, -1,  1], dtype=torch.int8)
        >>> quantize_bipolar(x, ternary=True, zero_thresh=0.2)
        tensor([-1,  0,  1,  1], dtype=torch.int8)
    """
    if ternary:
        zt = zero_thresh if zero_thresh is not None else abs(threshold)
        pos_mask = input > abs(threshold)
        neg_mask = input < -abs(zt)

        result = torch.zeros_like(input, dtype=torch.int8)
        result[pos_mask] = 1
        result[neg_mask] = -1
        return result
    else:
        return torch.where(input > threshold, torch.tensor(1, dtype=torch.int8),
                           torch.tensor(-1, dtype=torch.int8))


def block_float_quantize(
    input: Tensor,
    bits: int,
    frac: int,
    block_size: int = 64,
    *,
    rounding: str = "nearest",
) -> QuantizedWeight:
    r"""Quantize with a shared exponent for every *block_size* elements.

    This is a hardware-friendly scheme: for each block of ``block_size`` elements
    compute a shared scale, quantize all elements in that block to the same fixed-point
    format, and store the exponent along each block.

    Args:
        input (Tensor): Floating-point tensor.
        bits (int): Total bit-width.
        frac (int): Fixed fractional bit allocation (the exponent varies per block).
        block_size (int): Number of elements sharing a scale factor.
        rounding (str, optional): Rounding mode.  Default: ``"nearest"``.

    Returns:
        QuantizedWeight: Quantized data with per-block scale factors in ``.scale``.
    """
    cfg = FixedPointConfig(bits=bits, frac=frac, signed=True, rounding=rounding)
    flat = input.flatten()
    n_el = flat.numel()

    # Pad to multiple of block_size
    pad = (block_size - n_el % block_size) % block_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    blocks = flat.reshape(-1, block_size)
    n_blocks = blocks.size(0)

    # Per-block scale: scale so that max abs value maps to full range
    abs_max = blocks.abs().max(dim=1, keepdim=True).values
    eps = 1e-12
    block_scales = (2.0 ** (cfg.int_bits - 1) - cfg.step) / (abs_max + eps)

    scaled_blocks = blocks * block_scales
    q_blocks = _clamp_and_round(scaled_blocks, cfg)

    q_flat = q_blocks.flatten()[:n_el]
    scale_flat = block_scales.flatten()

    return QuantizedWeight(q_flat.reshape(input.shape), cfg, input.shape, scale=scale_flat)


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

def pack_bits(
    data: Tensor,
    bits_per_element: int,
    order: Literal["big", "little"] = "big",
) -> Tensor:
    r"""Pack integer codes into a compact bit-array.

    Each element of ``data`` is assumed to fit in ``bits_per_element`` bits.
    Elements are packed sequentially into ``torch.uint8`` bytes.

    This is the final step before writing to a hardware memory image.

    Args:
        data (LongTensor): Integer codes to pack.
        bits_per_element (int): Number of bits per element (e.g. 1, 2, 4, 8).
        order (str, optional): ``"big"`` (MSB first) or ``"little"`` (LSB first).

    Returns:
        LongTensor: Packed bytes with dtype ``torch.int64`` (values 0–255).
            Use ``.to(torch.uint8)`` to get raw bytes.

    Examples::

        >>> vals = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], dtype=torch.long)
        >>> pack_bits(vals, bits_per_element=1, order="big")
        tensor([180])   # 0b10110100
    """
    if bits_per_element > 32:
        raise ValueError(f"bits_per_element must be <= 32, got {bits_per_element}")

    flat = data.flatten()
    total_bits = flat.numel() * bits_per_element
    total_bytes = (total_bits + 7) // 8

    # Work in uint64 for accumulation
    if order == "big":
        # MSB-first: first element goes into most significant bits
        packed = torch.zeros(total_bytes, dtype=torch.int64)
        for i, val in enumerate(flat):
            val_int = int(val) & ((1 << bits_per_element) - 1)
            bit_pos = i * bits_per_element
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8

            bits_remaining = bits_per_element
            val_shifted = val_int

            while bits_remaining > 0:
                space = 8 - bit_offset
                take = min(bits_remaining, space)
                # Put the top `take` bits of val_shifted into the byte
                shift = bits_remaining - take
                chunk = (val_shifted >> shift) & ((1 << take) - 1)
                packed[byte_idx] |= chunk << (space - take)

                bits_remaining -= take
                bit_offset = 0
                byte_idx += 1
                val_shifted = val_shifted & ((1 << bits_remaining) - 1)

        return packed
    else:
        # LSB-first: first element goes into least significant bits
        packed = torch.zeros(total_bytes, dtype=torch.int64)
        for i, val in enumerate(flat):
            val_int = int(val) & ((1 << bits_per_element) - 1)
            bit_pos = i * bits_per_element
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8

            bits_remaining = bits_per_element
            val_shifted = val_int

            while bits_remaining > 0:
                space = 8 - bit_offset
                take = min(bits_remaining, space)
                # Put the bottom `take` bits of val_shifted into the byte
                chunk = val_shifted & ((1 << take) - 1)
                packed[byte_idx] |= chunk << bit_offset

                bits_remaining -= take
                bit_offset = 0
                byte_idx += 1
                val_shifted >>= take

        return packed


def unpack_bits(
    packed: Tensor,
    num_elements: int,
    bits_per_element: int,
    order: Literal["big", "little"] = "big",
) -> Tensor:
    r"""Unpack a bit-packed array back to integer codes.

    Inverse of :func:`pack_bits`.

    Args:
        packed (LongTensor): Packed bytes (values 0–255).
        num_elements (int): Number of elements to extract.
        bits_per_element (int): Number of bits per element.
        order (str): Byte order used during packing.

    Returns:
        LongTensor: Recovered integer codes.
    """
    result = torch.zeros(num_elements, dtype=torch.int64)

    for i in range(num_elements):
        bit_pos = i * bits_per_element
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        val = 0
        bits_remaining = bits_per_element

        while bits_remaining > 0:
            space = 8 - bit_offset
            take = min(bits_remaining, space)

            if order == "big":
                chunk = (packed[byte_idx] >> (space - take)) & ((1 << take) - 1)
                val = (val << take) | chunk
            else:
                chunk = (packed[byte_idx] >> bit_offset) & ((1 << take) - 1)
                val = val | (chunk << (bits_remaining - take))

            bits_remaining -= take
            bit_offset = 0
            byte_idx += 1

        result[i] = val

    return result


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

def export_verilog(
    qw: QuantizedWeight,
    filepath: str,
    *,
    module_name: str = "hd_weights",
    radix: int = 16,
    words_per_line: int = 8,
) -> None:
    r"""Export quantized weights as a Verilog ``localparam`` memory array.

    The output is a synthesizable SystemVerilog snippet:

    .. code-block:: systemverilog

        localparam logic [7:0] hd_weights [0:1535] = '{
            8'hA3, 8'h4F, 8'h12, ...
        };

    Args:
        qw (QuantizedWeight): Quantized weight data.
        filepath (str): Output ``.sv`` file path.
        module_name (str): Name of the parameter array.
        radix (int): Radix for integer literals (usually 16 for hex).
        words_per_line (int): Number of hex words per line.
    """
    flat = qw.data.flatten()
    bits = qw.config.bits

    # Determine word width in bytes
    word_bytes = (bits + 7) // 8

    lines = []
    lines.append("// Auto-generated fixed-point weight array")
    lines.append(f"// Format: {qw.config!r}, scale = {qw.scale!r}")
    lines.append(f"// Original shape: {tuple(qw.original_shape)}")
    lines.append(f"localparam logic [{bits-1}:0] {module_name} [0:{flat.numel()-1}] = '{{")

    idx = 0
    while idx < flat.numel():
        chunk = flat[idx:idx + words_per_line]
        vals = []
        for v in chunk:
            v_int = int(v) & ((1 << bits) - 1)
            if radix == 16:
                fmt = f"{bits}'h{{v_int:0{(bits+3)//4}x}}"
            elif radix == 10:
                fmt = f"{bits}'d{v_int}"
            elif radix == 2:
                fmt = f"{bits}'b{{v_int:0{bits}b}}"
            else:
                fmt = str(v_int)
            vals.append(fmt)
        suffix = "," if idx + words_per_line < flat.numel() else ""
        lines.append("    " + ", ".join(vals) + suffix)
        idx += words_per_line

    lines.append("};")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_c_header(
    qw: QuantizedWeight,
    filepath: str,
    *,
    array_name: str = "hd_weights",
    const: bool = True,
) -> None:
    r"""Export quantized weights as a C header file.

    Produces a ``const`` (or non-const) C array:

    .. code-block:: c

        // Auto-generated fixed-point weight array
        // Format: Q3.4s, scale = 1.000000
        const int8_t hd_weights[1536] = {
            -53, 79, 12, ...
        };

    Args:
        qw (QuantizedWeight): Quantized weight data.
        filepath (str): Output ``.h`` file path.
        array_name (str): C array variable name.
        const (bool): Whether to declare the array as ``const``.
    """
    flat = qw.data.flatten()
    bits = qw.config.bits
    signed = qw.config.signed

    # Choose a C integer type that fits
    if bits <= 8:
        ctype = "int8_t"
    elif bits <= 16:
        ctype = "int16_t"
    elif bits <= 32:
        ctype = "int32_t"
    else:
        ctype = "int64_t"

    if not signed:
        ctype = "u" + ctype

    sign_ext = signed and bits < 32

    lines = []
    lines.append("// Auto-generated fixed-point weight array")
    lines.append(f"// Format: {qw.config!r}, scale = {qw.scale!r}")
    lines.append(f"// Original shape: {tuple(qw.original_shape)}")
    lines.append(f"#include <stdint.h>")
    lines.append("")

    qualifier = "const " if const else ""
    lines.append(f"{qualifier}{ctype} {array_name}[{flat.numel()}] = {{")

    words_per_line = 16
    idx = 0
    while idx < flat.numel():
        chunk = flat[idx:idx + words_per_line]
        vals = []
        for v in chunk:
            v_int = int(v) & ((1 << bits) - 1)
            if sign_ext and v_int >= (1 << (bits - 1)):
                v_int = v_int - (1 << bits)
            vals.append(str(v_int))
        suffix = "," if idx + words_per_line < flat.numel() else ""
        lines.append("    " + ", ".join(vals) + suffix)
        idx += words_per_line

    lines.append("};")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_binary(qw: QuantizedWeight, filepath: str) -> None:
    r"""Export quantized weights as a raw binary file (compact memory image).

    Writes packed bytes in little-endian order.  For single-bit (bipolar) weights
    the data is packed to 1 bit per element, MSB-first within each byte.

    Args:
        qw (QuantizedWeight): Quantized weight data.
        filepath (str): Output ``.bin`` file path.

    Notes:
        Stored as: ``[<raw bytes of packed array>]``.  No header, no separator.
    """
    bits = qw.config.bits

    if bits <= 8:
        # Direct byte packing
        flat = qw.data.flatten()
        mask = (1 << bits) - 1
        vals = (flat & mask).to(torch.uint8)
        raw = vals.numpy().tobytes()
    else:
        # Multi-byte: pack using struct
        flat = qw.data.flatten()
        mask = (1 << bits) - 1
        vals = flat & mask
        word_bytes = (bits + 7) // 8
        fmt_str = ">" if word_bytes > 1 else "B"
        if word_bytes == 1:
            fmt_str = "B"
        elif word_bytes == 2:
            fmt_str = "<H"
        elif word_bytes <= 4:
            fmt_str = "<I"
        else:
            fmt_str = "<Q"
        raw = b""
        for v in vals:
            raw += struct.pack(fmt_str, int(v))

    with open(filepath, "wb") as f:
        f.write(raw)