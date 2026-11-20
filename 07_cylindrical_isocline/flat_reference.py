"""A flat reference layer-cake with any facing, for example 07's restore event.

Lives in a module, not in the fitting notebook, so the saved model (``restore.pt``)
can be loaded by the meshing notebook. Needs curlew and torch.
"""
import numpy as np
import torch

import curlew
from curlew.kinematics.restoration import _FlatReferenceStratigraphy


class FlatReference(_FlatReferenceStratigraphy):
    """A flat layer-cake with unit normal ``normal`` (pointing to younger beds): φ = normal · x.

    curlew's default flat reference with any facing; with a normal perpendicular to the
    hinge, a cylindrical restore stays cylindrical. ``depth_axis`` is the axis nearest the
    normal.
    """

    __slots__ = ("normal",)

    def __init__(self, normal):
        n = np.asarray(normal, dtype=float)
        self.normal = torch.as_tensor(n / np.linalg.norm(n), dtype=curlew.dtype, device=curlew.device)
        super().__init__(int(self.normal.abs().argmax()))

    def gradient(self, x):
        n = self.normal.to(dtype=x.dtype, device=x.device)
        return n.expand_as(x).clone()

    def value_and_gradient(self, x):
        n = self.normal.to(dtype=x.dtype, device=x.device)
        return x @ n, n.expand_as(x).clone()

    def __repr__(self):
        return f"FlatReference(normal={self.normal.tolist()})"
