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
r"""Golden trace generation and SystemVerilog testbench export.

This module captures the cycle-by-cycle behavior of a torchhd model and
renders it into synthesizable SystemVerilog testbench components.  The
workflow is:

1. Wrap your model in :class:`GoldenTrace`.
2. Run your training or inference loop — every ``step``, ``add``, or encoding
   call is recorded with its inputs and outputs.
3. Call :meth:`GoldenTrace.write_sv_stimulus` to emit an ``initial`` block
   that replays every input transition at the correct cycle offset.
4. Call :meth:`GoldenTrace.write_sv_checker` to emit a parallel checker block
   with ``assert`` statements comparing RTL outputs to golden expected values.

Classes
-------
GoldenTrace
    Cycle-accurate log of model I/O for testbench generation.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import math

import torch
from torch import Tensor


__all__ = [
    "GoldenTrace",
]


class GoldenTrace:
    r"""Cycle-accurate trace of model inputs and expected outputs.

    Records every operation performed on a wrapped model: encoding calls,
    weight updates, forward passes.  Produces a timestamped log that can be
    rendered into SystemVerilog stimulus/checker blocks.

    Args:
        name: Module display name for generated comments.
        clk_period_ns: Clock period for the stimulus file in nanoseconds.
            Default: 10 (100 MHz).
        tolerance: Acceptable absolute error between RTL output and golden
            output for floating-point comparisons.  For integer/fixed-point
            comparisons tolerance is in LSBs.

    Attributes:
        events (list): List of recorded events.  Each event is a dict with:
            ``cycle``, ``kind``, ``input_data``, ``output_data``, ``metadata``.
    """

    name: str
    clk_period_ns: float
    tolerance: float
    events: List[Dict[str, Any]]

    def __init__(
        self,
        name: str = "hd_model",
        clk_period_ns: float = 10.0,
        tolerance: float = 1e-6,
    ) -> None:
        self.name = name
        self.clk_period_ns = clk_period_ns
        self.tolerance = tolerance
        self.events = []
        self._cycle = 0

    def _record(
        self,
        kind: str,
        inputs: Dict[str, Any],
        outputs: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one event."""
        event = {
            "cycle": self._cycle,
            "kind": kind,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
        }
        self.events.append(event)

    def record_encode(
        self,
        input_data: Tensor,
        encoded: Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Record an encoding step (input → hypervector).

        Args:
            input_data: Raw input features.
            encoded: Resulting hypervector(s).
            metadata: Optional extra info (feature names, etc.).
        """
        self._record(
            "encode",
            {"input": input_data.detach().clone()},
            encoded.detach().clone(),
            metadata,
        )
        self._cycle += 1

    def record_step(
        self,
        input_hv: Tensor,
        target: Any,
        outputs: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Record a training step (hypervector + label → update).

        Args:
            input_hv: Input hypervector.
            target: Class label or target vector.
            outputs: Model output logits (if prediction was made).
            metadata: Optional extra info.
        """
        if isinstance(target, torch.Tensor):
            tgt_copy = target.detach().clone()
        elif isinstance(target, (int, float)):
            tgt_copy = target
        else:
            tgt_copy = str(target)

        self._record(
            "step",
            {"input": input_hv.detach().clone(), "target": tgt_copy},
            outputs.detach().clone() if outputs is not None and isinstance(outputs, torch.Tensor) else outputs,
            metadata,
        )
        self._cycle += 1

    def record_forward(
        self,
        input_hv: Tensor,
        outputs: Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Record an inference forward pass.

        Args:
            input_hv: Input hypervector.
            outputs: Model output logits or class scores.
            metadata: Optional extra info.
        """
        self._record(
            "forward",
            {"input": input_hv.detach().clone()},
            outputs.detach().clone(),
            metadata,
        )
        self._cycle += 1

    def record_custom(
        self,
        kind: str,
        inputs: Dict[str, Any],
        outputs: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Record a custom event with arbitrary I/O.

        Args:
            kind: Event type label.
            inputs: Dict of input name → tensor/scalar.
            outputs: Output tensor or scalar.
            metadata: Optional extra info.
        """
        inputs_clean = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_clean[k] = v.detach().clone()
            else:
                inputs_clean[k] = v

        if isinstance(outputs, torch.Tensor):
            outputs_clean = outputs.detach().clone()
        else:
            outputs_clean = outputs

        self._record(kind, inputs_clean, outputs_clean, metadata)
        self._cycle += 1

    # ----------------------------------------------------------------
    # SystemVerilog export
    # ----------------------------------------------------------------

    def write_sv_stimulus(
        self,
        filepath: str,
        *,
        input_bits: int = 8,
        output_bits: int = 32,
        signal_map: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""Emit a SystemVerilog stimulus ``initial`` block.

        Generates an ``initial begin ... end`` block that replays every
        recorded input at the corresponding cycle boundary using ``#CLK_PERIOD``
        delays.  The resulting file can be ``\`include``-d into a testbench.

        Args:
            filepath: Output ``.sv`` file path.
            input_bits: Bit-width of input signals (used for formatting).
            output_bits: Bit-width of output signals.
            signal_map: Optional mapping from trace signal names to RTL
                signal names (e.g. ``{"input": "hv_in", "target": "class_in"}``).

        Example output::

            // Stimulus for hd_model
            // Cycle 0: encode
            hv_in = 256'hA3F4...;
            #10;  // wait for clock
            // Cycle 1: step
            hv_in = 256'h12C0...;
            class_in = 8'd3;
            #10;
        """
        if signal_map is None:
            signal_map = {
                "input": "hv_in",
                "target": "class_in",
            }

        lines = []
        lines.append(f"// Auto-generated stimulus for {self.name}")
        lines.append(f"// {len(self.events)} events, clk_period = {self.clk_period_ns} ns")
        lines.append("")
        lines.append(f"initial begin")
        lines.append(f"  // Reset / wait for reset")
        lines.append(f"  #({self.clk_period_ns} * 2);")
        lines.append("")

        for i, event in enumerate(self.events):
            kind = event["kind"]
            cycle = event["cycle"]
            lines.append(f"  // Cycle {cycle} ({kind})")

            # Emit input assignments
            for iname, ivalue in event["inputs"].items():
                sig = signal_map.get(iname, iname)
                if isinstance(ivalue, torch.Tensor):
                    flat = ivalue.flatten()
                    if flat.numel() == 1:
                        val = int(flat[0]) & ((1 << input_bits) - 1)
                        lines.append(f"  {sig} = {input_bits}'d{val};")
                    else:
                        # Multi-element vector: emit as concatenation
                        hex_str = ""
                        for v in flat:
                            hex_str += f"{int(v) & ((1 << input_bits) - 1):02x}"
                        # SystemVerilog hex literal
                        bit_width = flat.numel() * input_bits
                        lines.append(f"  {sig} = {bit_width}'h{hex_str};")
                elif isinstance(ivalue, int):
                    lines.append(f"  {sig} = {input_bits}'d{ivalue};")
                else:
                    lines.append(f"  {sig} = {ivalue};")

            lines.append(f"  #({self.clk_period_ns});  // wait for clock")
            lines.append("")

        lines.append(f"  // Done")
        lines.append(f"  #({self.clk_period_ns} * 5);")
        lines.append(f"  $finish;")
        lines.append(f"end")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_sv_checker(
        self,
        filepath: str,
        *,
        input_bits: int = 8,
        output_bits: int = 32,
        signal_map: Optional[Dict[str, str]] = None,
        tolerance_lsb: int = 1,
    ) -> None:
        r"""Emit a SystemVerilog checker block with ``assert`` statements.

        For each recorded event that has output data, generates an assertion
        comparing the RTL output signal to the golden expected value.

        Args:
            filepath: Output ``.sv`` file path.
            input_bits: Bit-width of input signals.
            output_bits: Bit-width of output signals.
            signal_map: Mapping from trace signal names to RTL signal names.
                Default maps ``"output"`` → ``"hv_out"``, ``"score"`` → ``"score_out"``.
            tolerance_lsb: Allowed LSB error for integer/fixed-point comparisons.

        Example output::

            // Checker for hd_model
            always @(posedge clk) begin
              if (check_cycle == 0) begin
                assert (hv_out == 256'hEXPECTED) else $error("Cycle 0 mismatch");
              end
              ...
            end
        """
        if signal_map is None:
            signal_map = {
                "output": "hv_out",
                "score": "score_out",
                "class": "class_out",
            }

        lines = []
        lines.append(f"// Auto-generated checker for {self.name}")
        lines.append(f"// {len(self.events)} events, tolerance = ±{tolerance_lsb} LSB")
        lines.append("")

        # Attempt to infer a sensible output signal name from the events
        output_signal = "hv_out"
        for event in self.events:
            if event["outputs"] is not None:
                if isinstance(event["outputs"], torch.Tensor) and event["outputs"].numel() > 1:
                    output_signal = signal_map.get("output", "hv_out")
                else:
                    output_signal = signal_map.get("score", "score_out")
                break

        lines.append(f"// Output signal: {output_signal}")
        lines.append("")

        # Counter
        lines.append(f"reg [31:0] check_cycle = 0;")
        lines.append(f"reg [31:0] check_errors = 0;")
        lines.append("")
        lines.append(f"always @(posedge clk) begin")
        lines.append(f"  if (!rst_n) begin")
        lines.append(f"    check_cycle <= 0;")
        lines.append(f"    check_errors <= 0;")
        lines.append(f"  end else begin")
        lines.append(f"    case (check_cycle)")

        for i, event in enumerate(self.events):
            if event["outputs"] is None:
                continue
            outputs = event["outputs"]
            if not isinstance(outputs, torch.Tensor):
                continue

            flat = outputs.flatten()
            if flat.numel() == 1:
                expected = int(flat[0])
                lines.append(f"      {i}: begin")
                lines.append(f"        if ({output_signal} < ({expected} - {tolerance_lsb}) || "
                             f"{output_signal} > ({expected} + {tolerance_lsb})) begin")
                lines.append(f'          $error("Cycle %0d: expected ~%0d, got %0d", '
                             f"check_cycle, {expected}, {output_signal});")
                lines.append(f"          check_errors <= check_errors + 1;")
                lines.append(f"        end")
                lines.append(f"      end")
            else:
                # Multi-element: emit concatenated expected
                hex_str = ""
                for v in flat:
                    hex_str += f"{int(v) & ((1 << output_bits) - 1):08x}"
                bit_width = flat.numel() * output_bits
                lines.append(f"      {i}: begin")
                lines.append(f"        if ({output_signal} !== {bit_width}'h{hex_str}) begin")
                lines.append(f'          $error("Cycle %0d: output mismatch", check_cycle);')
                lines.append(f"          check_errors <= check_errors + 1;")
                lines.append(f"        end")
                lines.append(f"      end")

        lines.append(f"      default: ;")
        lines.append(f"    endcase")
        lines.append(f"    check_cycle <= check_cycle + 1;")
        lines.append(f"  end")
        lines.append(f"end")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the trace."""
        kinds = {}
        total_inputs = 0
        total_outputs = 0
        for e in self.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            if e["outputs"] is not None:
                total_outputs += 1

        lines = [
            f"GoldenTrace({self.name!r}): {len(self.events)} events over {self._cycle} cycles",
            f"  Clock: {self.clk_period_ns} ns",
            f"  Tolerance: {self.tolerance}",
        ]
        for kind, count in sorted(kinds.items()):
            lines.append(f"  {kind}: {count}")
        return "\n".join(lines)