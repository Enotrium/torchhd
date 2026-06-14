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
r"""Streaming and online learning primitives for hyperdimensional computing.

This module provides streaming-capable components that update model state
one sample at a time — no batching required.  The central building block is
the Hebbian accumulator, which maintains a running outer-product weight matrix
updated via a local Hebbian rule.

Classes
-------
HebbianAccumulator
    Running weight accumulation via outer-product Hebbian updates.
StreamingCentroid
    Online centroid classifier with Hebbian weight updates, one sample per step.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init

import torchhd.functional as functional

__all__ = [
    "HebbianAccumulator",
    "StreamingCentroid",
]


class HebbianAccumulator(nn.Module):
    r"""Running outer-product Hebbian accumulator for online hypervector learning.

    Maintains a weight matrix :math:`W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}`
    that is updated via the Hebbian outer product:

    .. math::
        W \leftarrow (1 - \eta) W + \eta \, (x^{\top} y)

    where :math:`x` is the input hypervector, :math:`y` is the target hypervector,
    and :math:`\eta \in [0, 1]` is the learning rate.  Setting :math:`\eta = 1`
    recovers standard additive (unweighted) accumulation; :math:`\eta < 1` implements
    an exponential moving average that gracefully forgets older samples — useful
    for non-stationary streams.

    The accumulator exposes a ``forward`` method for reading from the weight
    matrix, and a ``step`` method for accepting a single (input, target) pair.

    Args:
        in_features (int): Dimensionality of the input hypervectors :math:`d_{\text{in}}`.
        out_features (int): Dimensionality of the target hypervectors :math:`d_{\text{out}}`.
        lr (float, optional): Hebbian learning rate :math:`\eta`. Default: ``1.0``.
        device (``torch.device``, optional): Desired device of the weight matrix.
        dtype (``torch.dtype``, optional): Desired data type of the weight matrix.
        requires_grad (bool, optional): If autograd should track the weight matrix. Default: ``False``.

    Shape:
        - Input: :math:`(d_{\text{in}})` or :math:`(*, d_{\text{in}})`
        - Target: :math:`(d_{\text{out}})` or :math:`(*, d_{\text{out}})`
        - Weight: :math:`(d_{\text{in}}, d_{\text{out}})`

    Attributes:
        weight: The accumulated weight matrix of shape :math:`(d_{\text{in}}, d_{\text{out}})`.
        lr: Hebbian learning rate.
        num_steps: Running count of ``step`` calls (read-only).

    Examples::

        >>> acc = HebbianAccumulator(512, 512, lr=0.1)
        >>> x = torchhd.random(1, 512)
        >>> y = torchhd.random(1, 512)
        >>> acc.step(x, y)
        >>> read = acc(x)
        >>> read.shape
        torch.Size([1, 512])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    lr: float
    num_steps: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 1.0,
        device=None,
        dtype=None,
        requires_grad: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        weight = torch.zeros(in_features, out_features, **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)
        self.register_buffer("num_steps", torch.zeros((), dtype=torch.long))

    def forward(self, input: Tensor) -> Tensor:
        r"""Read from the accumulated weight matrix.

        .. math::
            \text{output} = x W

        Args:
            input (Tensor): Input hypervector(s) of shape :math:`(*, d_{\text{in}})`.

        Returns:
            Tensor: Read-result of shape :math:`(*, d_{\text{out}})`.
        """
        return input @ self.weight

    @torch.no_grad()
    def step(self, input: Tensor, target: Tensor) -> None:
        r"""Update the weight matrix with a single (input, target) pair via
        the Hebbian outer-product rule.

        .. math::
            W \leftarrow (1 - \eta) W + \eta \, (x^{\top} y)

        Accepts batched inputs for efficiency, though the intended use-case
        is one sample at a time.

        Args:
            input (Tensor): Input hypervector(s) of shape :math:`(*, d_{\text{in}})`.
            target (Tensor): Target hypervector(s) of shape :math:`(*, d_{\text{out}})`.
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # Hebbian outer product averaged over the batch
        hebbian_update = input.T @ target
        batch_size = input.size(0)

        if self.lr == 1.0:
            self.weight.add_(hebbian_update)
        else:
            # Exponential moving average: W = (1 - lr) * W + lr * (x^T y)
            self.weight.mul_(1.0 - self.lr).add_(hebbian_update, alpha=self.lr)

        self.num_steps.add_(batch_size)

    @torch.no_grad()
    def step_adaptive(
        self,
        input: Tensor,
        target: Tensor,
        pred: Optional[Tensor] = None,
    ) -> None:
        r"""Hebbian update with an anti-Hebbian correction for mispredicted samples.

        Implements a local error-driven learning rule:

        .. math::
            W \leftarrow (1 - \eta) W + \eta \left[ x^{\top} y_{\text{true}} - x^{\top} y_{\text{pred}} \right]

        When ``pred`` is ``None``, this degenerates to :meth:`step`.

        Args:
            input (Tensor): Input hypervector(s) :math:`(*, d_{\text{in}})`.
            target (Tensor): True target hypervector(s) :math:`(*, d_{\text{out}})`.
            pred (Tensor, optional): Predicted target hypervector(s) of same shape as ``target``.
                If provided, the anti-Hebbian term is subtracted.
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        hebbian = input.T @ target

        if pred is not None:
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            anti_hebbian = input.T @ pred
            hebbian = hebbian - anti_hebbian

        if self.lr == 1.0:
            self.weight.add_(hebbian)
        else:
            self.weight.mul_(1.0 - self.lr).add_(hebbian, alpha=self.lr)

        self.num_steps.add_(input.size(0))

    @torch.no_grad()
    def normalize_(self, eps: float = 1e-12) -> None:
        r"""Normalize each column of the weight matrix to unit length in-place.

        This is typically called after accumulation is complete, before switching
        to a dot-product readout.

        Args:
            eps (float): Epsilon for numerical stability.
        """
        norms = self.weight.norm(dim=0, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

    @torch.no_grad()
    def reset_(self) -> None:
        """Zero the weight matrix and reset the step counter."""
        self.weight.zero_()
        self.num_steps.zero_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"lr={self.lr}, num_steps={self.num_steps.item()}"
        )


class StreamingCentroid(nn.Module):
    r"""Online centroid classifier that processes one sample per ``step`` call.

    This is the streaming equivalent of :class:`~torchhd.models.Centroid`.  Under
    the hood it maintains a :class:`HebbianAccumulator` that maps input hypervectors
    to an ``out_features``-dimensional class space.  Each class is represented as a
    one-hot vector, so the Hebbian outer product naturally accumulates a class-prototype
    matrix.

    The module supports both **accumulation** (prototypes are sums of all class
    samples) and **exponential moving average** (``lr < 1``) modes.

    Args:
        in_features (int): Dimensionality of the input hypervectors.
        out_features (int): Number of output classes.
        lr (float, optional): Hebbian learning rate :math:`\eta`. Default: ``1.0``.
        device (``torch.device``, optional): Desired device.
        dtype (``torch.dtype``, optional): Desired data type.
        requires_grad (bool, optional): If autograd should track parameters. Default: ``False``.

    Shape:
        - Input: :math:`(d_{\text{in}})` or :math:`(*, d_{\text{in}})`
        - Output: :math:`(1, \text{out\_features})` or :math:`(*, \text{out\_features})`

    Attributes:
        accumulator: The underlying :class:`HebbianAccumulator`.
        prototype_weight: Convenience view of the accumulated class-prototype matrix
            of shape :math:`(\text{out\_features}, d_{\text{in}})`.

    Examples::

        >>> sc = StreamingCentroid(512, 10)
        >>> x = torchhd.random(1, 512)
        >>> sc.step(x, 3)          # label for class 3
        >>> sc.step(x, 7)          # label for class 7
        >>> logits = sc(x)         # dot-product similarity to all classes
        >>> logits.shape
        torch.Size([1, 10])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 1.0,
        device=None,
        dtype=None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Hebbian accumulator maps input -> one-hot class vector.
        # Accumulator weight shape: (in_features, out_features).
        self.accumulator = HebbianAccumulator(
            in_features=in_features,
            out_features=out_features,
            lr=lr,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @property
    def prototype_weight(self) -> Tensor:
        r"""Class-prototype matrix of shape :math:`(n_{\text{classes}}, d)`.

        This is the transpose of the accumulator weight, providing a natural
        ``(n_classes, d)`` view for dot-product similarity comparisons.
        """
        return self.accumulator.weight.T

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        r"""Compute per-class similarity scores.

        Args:
            input (Tensor): Input hypervector(s) :math:`(*, d)`.
            dot (bool, optional): If ``True``, use dot-product similarity instead
                of cosine similarity.  Faster after :meth:`normalize` has been called.

        Returns:
            Tensor: Similarity scores of shape :math:`(*, n_{\text{classes}})`.
        """
        prototypes = self.prototype_weight
        if dot:
            return functional.dot_similarity(input, prototypes)
        return functional.cosine_similarity(input, prototypes)

    @torch.no_grad()
    def step(self, input: Tensor, target: Union[int, Tensor], lr: Optional[float] = None) -> None:
        r"""Process a single sample (or small batch) with its class label.

        Args:
            input (Tensor): Input hypervector(s) :math:`(*, d)`.
            target (int or Tensor): Class label(s).  If an integer is given it is
                treated as a scalar class index.  If a tensor of integers is given
                each element is treated as a separate label.
            lr (float, optional): Per-step learning rate override.  If ``None``,
                uses the accumulator's default.
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        # Build one-hot target vector(s)
        device = input.device
        dtype = self.accumulator.weight.dtype
        n = input.size(0)

        if isinstance(target, int):
            target = torch.tensor([target], device=device, dtype=torch.long)

        one_hot = torch.zeros(n, self.out_features, device=device, dtype=dtype)
        one_hot[torch.arange(n, device=device), target] = 1.0

        # Hebbian step: outer product of input and one-hot target
        prev_lr = self.accumulator.lr
        if lr is not None:
            self.accumulator.lr = float(lr)
        try:
            self.accumulator.step(input, one_hot)
        finally:
            self.accumulator.lr = prev_lr

    @torch.no_grad()
    def step_online(
        self, input: Tensor, target: Union[int, Tensor], lr: Optional[float] = None
    ) -> None:
        r"""OnlineHD-style adaptive update: only updates prototypes when prediction is wrong.

        Implements the single-pass online training rule from `OnlineHD: Robust, Efficient,
        and Single-Pass Online Learning Using Hyperdimensional System
        <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        The prediction is made *before* the Hebbian update so that each sample is
        evaluated fairly (test-then-train).

        Args:
            input (Tensor): Input hypervector(s) :math:`(*, d)`.
            target (int or Tensor): True class label(s).
            lr (float, optional): Per-step learning rate override.
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        device = input.device
        dtype = self.accumulator.weight.dtype
        n = input.size(0)

        if isinstance(target, int):
            target = torch.tensor([target], device=device, dtype=torch.long)

        # --- Predict before updating (test-then-train) ---
        prototypes = self.prototype_weight
        logits = functional.cosine_similarity(input, prototypes)
        pred = logits.argmax(dim=1)

        # Identify mistakes
        is_wrong = target != pred
        if is_wrong.sum().item() == 0:
            self.accumulator.num_steps.add_(n)
            return

        wrong_input = input[is_wrong]
        wrong_target = target[is_wrong]
        wrong_pred = pred[is_wrong]

        m = wrong_input.size(0)

        one_hot_target = torch.zeros(m, self.out_features, device=device, dtype=dtype)
        one_hot_target[torch.arange(m, device=device), wrong_target] = 1.0

        one_hot_pred = torch.zeros(m, self.out_features, device=device, dtype=dtype)
        one_hot_pred[torch.arange(m, device=device), wrong_pred] = 1.0

        # Alpha scaling: (1 - cos(x, pred_cls)) and (cos(x, true_cls) - 1)
        alpha_target = 1.0 - logits[is_wrong].gather(1, wrong_target.unsqueeze(1))
        alpha_pred = logits[is_wrong].gather(1, wrong_pred.unsqueeze(1)) - 1.0

        hebbian = (
            wrong_input.T @ (alpha_target * one_hot_target)
            + wrong_input.T @ (alpha_pred * one_hot_pred)
        )

        prev_lr = self.accumulator.lr
        if lr is not None:
            self.accumulator.lr = float(lr)

        try:
            if self.accumulator.lr == 1.0:
                self.accumulator.weight.add_(hebbian, alpha=lr if lr is not None else 1.0)
            else:
                # Apply per-step scaling: W = (1-lr)*W + lr*(hebbian)
                self.accumulator.weight.mul_(1.0 - self.accumulator.lr).add_(
                    hebbian, alpha=self.accumulator.lr * (lr if lr is not None else 1.0)
                )
        finally:
            self.accumulator.lr = prev_lr

        self.accumulator.num_steps.add_(n)

    @torch.no_grad()
    def normalize(self, eps: float = 1e-12) -> None:
        r"""Normalize all class prototypes to unit length in-place.

        After calling this, the ``forward`` pass can use ``dot=True`` for
        faster inference (dot-product instead of cosine similarity).
        """
        prototypes = self.prototype_weight
        norms = prototypes.norm(dim=1, keepdim=True)
        if torch.isclose(norms, torch.zeros_like(norms), equal_nan=True).any():
            import warnings
            warnings.warn(
                "The norm of a prototype vector is nearly zero upon normalizing, "
                "this could indicate a bug."
            )
        norms.clamp_(min=eps)
        self.accumulator.weight.div_(norms.T)

    @torch.no_grad()
    def reset(self) -> None:
        """Reset all prototypes to zero and clear the step counter."""
        self.accumulator.reset_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"lr={self.accumulator.lr}, num_steps={self.accumulator.num_steps.item()}"
        )