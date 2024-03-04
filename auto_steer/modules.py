import math
from typing import Union

import numpy as np
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class TranslationTransform(nn.Module):
    """
    A nn.Module that applies a translation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.translation = t.nn.Parameter(t.zeros(d_model))

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the translation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return x + self.translation


class MeanTranslationTransform(nn.Module):
    """
    A nn.Module that applies a mean translation transformation to the input tensor.

    Args:
        mean_diff (Tensor): The mean difference tensor.

    Attributes:
        mean_diff (Tensor): The mean difference tensor.
    """

    def __init__(self, mean_diff: Float[Tensor, "d_model"]):
        super().__init__()
        self.mean_diff = mean_diff

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the mean translation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return x + self.mean_diff


class UncenteredLinearMapTransform(nn.Module):
    """
    A nn.Module that applies an uncentered linear map transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        linear_map (t.nn.Linear): The linear map matrix.
        center (t.nn.Parameter): A learned "center".
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear_map = t.nn.Linear(d_model, d_model, bias=False)
        self.center = nn.Parameter(t.empty(d_model))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_map.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.center, -bound, bound)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the uncentered linear map transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.linear_map(x + self.center) - self.center


class RotationTransform(nn.Module):
    """
    A nn.Module that applies a rotation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.rotation_pre = nn.Linear(d_model, d_model, bias=False)
        self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)

    def forward(
        self, x: Union[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_model"]]
    ) -> Union[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_model"]]:
        """
        Applies the rotation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.rotation(x)


class OffsetRotationTransform(nn.Module):
    """
    A nn.Module that applies an offset rotation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        offset (t.nn.Parameter): The offset vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.rotation_pre = t.nn.Linear(d_model, d_model, bias=False)
        self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        self.offset = nn.Parameter(t.empty(d_model))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rotation.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.offset, -bound, bound)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the offset rotation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.rotation(x) + self.offset


class UncenteredRotationTransform(nn.Module):
    """
    A nn.Module that applies an uncentered rotation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.rotation_pre = t.nn.Linear(d_model, d_model, bias=False)
        self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        self.center = nn.Parameter(t.empty(d_model))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rotation.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.center, -bound, bound)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the uncentered rotation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.rotation(x + self.center) - self.center
