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
r"""Basis, codebook, and memory-initialization export for RTL synthesis.

This module bridges the gap between torchhd's floating-point hypervector
representations and the initialization files that FPGA/ASIC memory generators
consume.  It extracts projection matrices, basis hypervectors, and codebook
mappings from torchhd embedding modules and renders them into:

* Xilinx COE format (``blk_mem_gen``)
* Altera/Intel MIF format
* Raw hex (``$readmemh`` for Verilog testbenches)
* BRAM depth×width ready hex images

Functions
---------
extract_basis
    Pull the basis hypervectors from an embedding module.
extract_codebook
    Pull the per-level/per-class ID mapping from an embedding.
extract_projection_matrix
    Extract the full (N_features, D) projection matrix.
export_coe
    Write a Xilinx COE file.
export_mif
    Write an Altera/Intel MIF file.
export_hex
    Write a raw hex file for ``$readmemh``.
export_bram_init
    Write a depth×width hex image ready for BRAM initialization.
"""

from typing import Optional, Dict, Union
import os

import torch
from torch import Tensor
import torch.nn as nn

from torchhd.quantize import QuantizedWeight, to_fixed_point, FixedPointConfig

__all__ = [
    "extract_basis",
    "extract_codebook",
    "extract_projection_matrix",
    "export_coe",
    "export_mif",
    "export_hex",
    "export_bram_init",
]


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_basis(embedding: nn.Module) -> Dict[str, Tensor]:
    r"""Extract basis hypervectors from a torchhd embedding module.

    Inspects the module's registered parameters and buffers to find the
    basis vectors (level hypervectors, random projections, etc.) that the
    RTL needs to replicate for encoding.

    Args:
        embedding: A torchhd embedding module (e.g. ``embeddings.Random``,
            ``embeddings.Level``, ``embeddings.Sinusoid``, ``embeddings.Density``).

    Returns:
        dict: Mapping of parameter names to tensors.

    Examples::

        >>> emb = torchhd.embeddings.Random(10, 512)
        >>> basis = extract_basis(emb)
        >>> list(basis.keys())
    """
    result = {}

    for name, param in embedding.named_parameters():
        result[f"param.{name}"] = param.data.detach().clone()

    for name, buf in embedding.named_buffers():
        result[f"buffer.{name}"] = buf.data.detach().clone()

    return result


def extract_codebook(embedding: nn.Module) -> Dict[int, Tensor]:
    r"""Extract the per-level or per-class ID mapping from an embedding.

    For level/thermometer/circular embeddings, this returns the mapping from
    each encoded value (0..N-1) to its corresponding hypervector.
    For random embeddings, this returns the projection matrix rows.

    Args:
        embedding: A torchhd embedding module.

    Returns:
        dict: Mapping ``{index: hypervector_tensor}``.

    Examples::

        >>> emb = torchhd.embeddings.Level(10, 512)
        >>> codebook = extract_codebook(emb)
        >>> len(codebook)
        10
    """
    basis = extract_basis(embedding)
    codebook = {}

    # Common patterns in torchhd embeddings:
    # - 'weight' parameter is often the projection matrix (num_levels x dims)
    # - Some embeddings store basis vectors as 'basis' or 'vectors'
    for key, tensor in basis.items():
        if tensor.dim() == 2:
            # (N, D) — treat as codebook entries
            for i in range(tensor.size(0)):
                codebook[i] = tensor[i].clone()
        elif tensor.dim() == 1:
            # (D,) — single entry
            codebook[len(codebook)] = tensor.clone()

    return codebook


def extract_projection_matrix(embedding: nn.Module) -> Tensor:
    r"""Extract the full (N_features, D) projection matrix from an embedding.

    Searches the module's parameters for a 2-D weight tensor that represents
    the projection from input features to hypervector dimensions.

    Args:
        embedding: A torchhd embedding module.

    Returns:
        Tensor: The projection matrix of shape ``(in_features, dimensions)``.

    Examples::

        >>> emb = torchhd.embeddings.Random(10, 512)
        >>> proj = extract_projection_matrix(emb)
        >>> proj.shape
        torch.Size([10, 512])
    """
    # Look for the largest 2-D weight tensor
    candidates = []
    for name, param in embedding.named_parameters():
        if param.data.dim() == 2:
            candidates.append((name, param.data.detach()))

    if not candidates:
        raise RuntimeError(
            f"No 2-D parameter found in embedding {type(embedding).__name__}. "
            f"Available parameters: {list(dict(embedding.named_parameters()).keys())}"
        )

    # Return the largest one (usually the projection matrix)
    candidates.sort(key=lambda x: x[1].numel(), reverse=True)
    return candidates[0][1].clone()


# ---------------------------------------------------------------------------
# COE format (Xilinx)
# ---------------------------------------------------------------------------

def export_coe(
    qw: QuantizedWeight,
    filepath: str,
    *,
    memory_name: str = "hd_memory",
    radix: int = 16,
    words_per_line: int = 8,
) -> None:
    r"""Export quantized weights as a Xilinx COE file.

    Produces a COE file suitable for the Xilinx ``blk_mem_gen`` IP core:

    .. code-block:: text

        ; Xilinx COE file for hd_memory
        ; Format: Q4.4s, scale=1.0
        memory_initialization_radix=16;
        memory_initialization_vector=
        A3, 4F, 12, C0, FF, 00, 7E, 81,
        ...

    Args:
        qw: Quantized weight data.
        filepath: Output ``.coe`` file path.
        memory_name: Name for the comment header.
        radix: Radix (2, 10, or 16).  Default: 16 (hex).
        words_per_line: Number of words per line.
    """
    flat = qw.data.flatten()
    bits = qw.config.bits
    n_words = flat.numel()

    radix_str = {2: "2", 10: "10", 16: "16"}.get(radix, "16")

    lines = []
    lines.append(f"; Xilinx COE file for {memory_name}")
    lines.append(f"; Format: {qw.config!r}, scale={qw.scale!r}")
    lines.append(f"; Original shape: {tuple(qw.original_shape)}")
    lines.append(f"; Depth: {n_words}, Width: {bits}")
    lines.append(f"memory_initialization_radix={radix_str};")
    lines.append(f"memory_initialization_vector=")

    idx = 0
    while idx < n_words:
        chunk = flat[idx:idx + words_per_line]
        vals = []
        for v in chunk:
            v_int = int(v) & ((1 << bits) - 1)
            if radix == 16:
                hex_digits = max(1, (bits + 3) // 4)
                vals.append(f"{v_int:0{hex_digits}x}")
            elif radix == 2:
                vals.append(f"{v_int:0{bits}b}")
            else:
                vals.append(f"{v_int}")
        suffix = "," if idx + words_per_line < n_words else ";"
        lines.append("  " + ", ".join(vals) + suffix)
        idx += words_per_line

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# MIF format (Altera/Intel)
# ---------------------------------------------------------------------------

def export_mif(
    qw: QuantizedWeight,
    filepath: str,
    *,
    memory_name: str = "hd_memory",
    radix: int = 16,
    words_per_line: int = 8,
) -> None:
    r"""Export quantized weights as an Altera/Intel MIF file.

    Produces a MIF file suitable for Intel/Altera memory IP:

    .. code-block:: text

        -- Altera MIF file for hd_memory
        DEPTH = 1536;
        WIDTH = 8;
        ADDRESS_RADIX = HEX;
        DATA_RADIX = HEX;
        CONTENT BEGIN
          000 : A3 4F 12 C0 FF 00 7E 81;
          008 : ...
        END;

    Args:
        qw: Quantized weight data.
        filepath: Output ``.mif`` file path.
        memory_name: Name for the comment header.
        radix: Data radix (10 or 16).  Default: 16.
        words_per_line: Words per line.
    """
    flat = qw.data.flatten()
    bits = qw.config.bits
    n_words = flat.numel()

    addr_radix = "HEX"
    data_radix = "HEX" if radix == 16 else "DEC"
    addr_width = max(1, (n_words - 1).bit_length())

    # For per-line addressing, we write `words_per_line` entries per line.
    # Each line covers `words_per_line` elements at sequential addresses.
    num_lines = (n_words + words_per_line - 1) // words_per_line
    line_addr_width = max(1, (num_lines - 1).bit_length())
    line_addr_digits = max(1, (line_addr_width + 3) // 4) if radix == 16 else 1

    lines = []
    lines.append(f"-- Altera MIF file for {memory_name}")
    lines.append(f"-- Format: {qw.config!r}, scale={qw.scale!r}")
    lines.append(f"DEPTH = {n_words};")
    lines.append(f"WIDTH = {bits};")
    lines.append(f"ADDRESS_RADIX = {addr_radix};")
    lines.append(f"DATA_RADIX = {data_radix};")
    lines.append("CONTENT BEGIN")

    idx = 0
    while idx < n_words:
        chunk = flat[idx:idx + words_per_line]
        vals = []
        for v in chunk:
            v_int = int(v) & ((1 << bits) - 1)
            if radix == 16:
                hex_digits = max(1, (bits + 3) // 4)
                vals.append(f"{v_int:0{hex_digits}x}")
            else:
                vals.append(f"{v_int}")
        if radix == 16:
            addr_str = f"{idx // words_per_line:0{line_addr_digits}x}"
        else:
            addr_str = f"{idx // words_per_line}"
        lines.append(f"  {addr_str} : {' '.join(vals)};")
        idx += words_per_line

    lines.append("END;")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Hex format (Verilog $readmemh)
# ---------------------------------------------------------------------------

def export_hex(
    qw: QuantizedWeight,
    filepath: str,
    *,
    radix: int = 16,
    words_per_line: int = 8,
) -> None:
    r"""Export quantized weights as a raw hex file for ``$readmemh``.

    Produces a plain hex file with one word per line (or space-separated),
    suitable for Verilog ``$readmemh`` or ``$readmemb``:

    .. code-block:: text

        // hex file for Verilog $readmemh
        a3
        4f
        12
        ...

    Args:
        qw: Quantized weight data.
        filepath: Output ``.hex`` file path.
        radix: 16 for ``$readmemh``, 2 for ``$readmemb``.
        words_per_line: Words per line (space-separated).
    """
    flat = qw.data.flatten()
    bits = qw.config.bits

    lines = []
    comment_char = "//" if radix == 2 else "//"
    lines.append(f"{comment_char} hex file for {'$readmemb' if radix == 2 else '$readmemh'}")

    idx = 0
    n_words = flat.numel()
    while idx < n_words:
        chunk = flat[idx:idx + words_per_line]
        vals = []
        for v in chunk:
            v_int = int(v) & ((1 << bits) - 1)
            if radix == 16:
                hex_digits = max(1, (bits + 3) // 4)
                vals.append(f"{v_int:0{hex_digits}x}")
            elif radix == 2:
                vals.append(f"{v_int:0{bits}b}")
            else:
                vals.append(f"{v_int}")
        lines.append(" ".join(vals))
        idx += words_per_line

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# BRAM initialization helper
# ---------------------------------------------------------------------------

def export_bram_init(
    qw: QuantizedWeight,
    filepath: str,
    *,
    bram_depth: Optional[int] = None,
    bram_width: int = 32,
    radix: int = 16,
    fill_value: int = 0,
) -> None:
    r"""Export quantized weights as a BRAM initialization image.

    Organizes the weight data into BRAM lines of ``bram_width`` bits,
    possibly spreading across multiple BRAM instances if the data is wider
    than one BRAM width.

    Args:
        qw: Quantized weight data.
        filepath: Output file path.
        bram_depth: Number of rows per BRAM.  If ``None``, uses the next
            power of 2 that fits the data.
        bram_width: Bits per BRAM word (default 32).
        radix: Output radix (16 = hex, 2 = binary).
        fill_value: Value to fill padding entries (default 0).

    Notes:
        If the element width exceeds ``bram_width``, multiple BRAM instances
        are written as separate sections in the file.  Each section is
        labeled ``// BRAM instance 0``, ``// BRAM instance 1``, etc.
    """
    flat = qw.data.flatten()
    elements = flat.numel()
    elem_bits = qw.config.bits

    # How many elements fit in one BRAM word?
    elements_per_word = bram_width // elem_bits
    if elements_per_word == 0:
        # Element is wider than BRAM word — split across multiple BRAMs
        brams_needed = (elem_bits + bram_width - 1) // bram_width
        elements_per_word = 1
    else:
        brams_needed = 1

    # Required depth
    words_needed = (elements + elements_per_word - 1) // elements_per_word

    if bram_depth is None:
        # Round up to next power of 2
        bram_depth = 1
        while bram_depth < words_needed:
            bram_depth *= 2
    elif bram_depth < words_needed:
        raise ValueError(
            f"bram_depth ({bram_depth}) is too small for "
            f"{words_needed} words needed ({elements} elements × "
            f"{elem_bits} bits / {bram_width} bits per word)"
        )

    with open(filepath, "w") as f:
        f.write(f"// BRAM initialization: {qw.original_shape}\n")
        f.write(f"// Format: {qw.config!r}, scale={qw.scale!r}\n")
        f.write(f"// Depth: {bram_depth}, Width: {bram_width}\n")
        f.write(f"// Instances: {brams_needed}\n\n")

        for instance in range(brams_needed):
            if brams_needed > 1:
                f.write(f"// BRAM instance {instance} (bits "
                        f"{instance * bram_width}.."
                        f"{min((instance + 1) * bram_width, elem_bits) - 1})\n")

            for row in range(bram_depth):
                word_val = 0
                for el_in_word in range(elements_per_word):
                    el_idx = row * elements_per_word + el_in_word
                    if el_idx < elements:
                        val = int(flat[el_idx]) & ((1 << elem_bits) - 1)

                        if brams_needed > 1:
                            # Extract the slice belonging to this BRAM instance
                            lsb = instance * bram_width
                            msb = min((instance + 1) * bram_width, elem_bits)
                            val = (val >> lsb) & ((1 << (msb - lsb)) - 1)
                            word_val = (word_val) | (val << 0)  # align to LSB of this slice
                        else:
                            word_val = (word_val << elem_bits) | val
                    else:
                        word_val = (word_val << (elem_bits if brams_needed == 1 else bram_width)) | fill_value

                if radix == 16:
                    hex_digits = bram_width // 4
                    # For multi-BRAM instances, the word is only as wide as the slice
                    if brams_needed > 1:
                        slice_width = min(bram_width, elem_bits)
                        hex_digits = (slice_width + 3) // 4
                    f.write(f"  {word_val:0{hex_digits}x}\n")
                else:
                    f.write(f"  {word_val:0{bram_width}b}\n")

            if instance < brams_needed - 1:
                f.write("\n")