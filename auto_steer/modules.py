import math
from typing import Optional, Union

import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from auto_steer.utils.misc import get_default_device

default_device = get_default_device()


class TranslationTransform(nn.Module):
    """
    A nn.Module that applies a translation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.

    Attributes:
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(
        self, d_model: int, device: Optional[Union[str, t.device]] = default_device
    ):
        super().__init__()
        self.translation = t.nn.Parameter(t.zeros(d_model, device=device))

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
        device (Optional[Union[str, t.device]]): The device on which the module should
        be initialised on.

    Attributes:
        mean_diff (Tensor): The mean difference tensor.
    """

    def __init__(
        self,
        mean_diff: Float[Tensor, "d_model"],
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.mean_diff = mean_diff.to(device)

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
    A nn.Module that applies an uncentered linear map transformation to the input
    tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        bias (bool): Whether to include a bias term.
        device (Optional[Union[str, t.device]]): The device on which the module should
        initialised on.

    Attributes:
        linear_map (t.nn.Linear): The linear map matrix.
        center (t.nn.Parameter): A learned "center".
    """

    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.linear_map = t.nn.Linear(d_model, d_model, bias=bias, device=device)
        self.center = nn.Parameter(t.empty(d_model, device=device))
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
        device (Optional[Union[str, t.device]]): The device on which module should be
        initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
    """

    def __init__(
        self, d_model: int, device: Optional[Union[str, t.device]] = default_device
    ):
        super().__init__()
        self.rotation_pre = nn.Linear(d_model, d_model, bias=False, device=device)
        # using orthogonal_map=cayley as it seems more performant
        self.rotation = t.nn.utils.parametrizations.orthogonal(
            self.rotation_pre, orthogonal_map="cayley"
        )
        # janky way to ensure that random initialisation of linear layer means that the
        # orthogonal matrix has a positive determinant (that there is no reflection
        # going on)
        while self.rotation.weight.clone().detach().det() < 0:
            self.rotation_pre = nn.Linear(d_model, d_model, bias=False, device=device)
            self.rotation = t.nn.utils.parametrizations.orthogonal(
                self.rotation_pre, orthogonal_map="cayley"
            )

    def forward(
        self, x: Union[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]
    ) -> Union[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        """
        Applies the rotation transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.rotation(x)


class BiasedRotationTransform(nn.Module):
    """
    A nn.Module that applies an rotation transformation plus a bias to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        device (Optional[Union[str, t.device]]): The device on which the module should
        be initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        bias (t.nn.Parameter): The bias vector.
    """

    def __init__(
        self, d_model: int, device: Optional[Union[str, t.device]] = default_device
    ):
        super().__init__()
        self.rotation_pre = t.nn.Linear(d_model, d_model, bias=False, device=device)
        self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        while self.rotation.weight.clone().detach().det() < 0:
            self.rotation_pre = nn.Linear(d_model, d_model, bias=False, device=device)
            self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        self.bias = nn.Parameter(t.empty(d_model, device=device))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rotation.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

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
        return self.rotation(x) + self.bias


class UncenteredRotationTransform(nn.Module):
    """
    A nn.Module that applies an uncentered rotation transformation to the input tensor.
    The uncentered rotation transformation adds a learned "center" to the input, rotates
    it via a learned rotation matrix, takes the learned center away and returns the
    result.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        device (Optional[Union[str, t.device]]): The device on which module should be
        initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(
        self, d_model: int, device: Optional[Union[str, t.device]] = default_device
    ):
        super().__init__()
        self.rotation_pre = t.nn.Linear(d_model, d_model, bias=False, device=device)
        self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        while self.rotation.weight.clone().detach().det() < 0:
            self.rotation_pre = nn.Linear(d_model, d_model, bias=False, device=device)
            self.rotation = t.nn.utils.parametrizations.orthogonal(self.rotation_pre)
        self.center = nn.Parameter(t.empty(d_model, device=device))
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


# losses ===============================================================================


class CosineSimilarityLoss(nn.Module):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return -nn.functional.cosine_similarity(predictions, targets, dim=-1).mean()


class L1CosineSimilarityLoss(nn.Module):
    def __init__(self, l1_lambda: float = 0.5):
        super().__init__()
        self.l1_lambda = l1_lambda

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        cosine_loss = -nn.functional.cosine_similarity(
            predictions, targets, dim=-1
        ).mean()
        l1_loss = nn.functional.l1_loss(predictions, targets).mean()
        return cosine_loss + self.l1_lambda * l1_loss


class L2CosineSimilarityLoss(nn.Module):
    def __init__(self, l2_lambda: float = 0.5):
        super().__init__()
        self.l2_lambda = l2_lambda

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        cosine_loss = -nn.functional.cosine_similarity(
            predictions, targets, dim=-1
        ).mean()
        l2_loss = nn.functional.mse_loss(predictions, targets).mean()
        return cosine_loss + self.l2_lambda * l2_loss


class MSELoss(nn.Module):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return nn.functional.mse_loss(predictions, targets)
