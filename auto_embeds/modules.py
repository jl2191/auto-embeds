import math
from typing import Optional, Union

import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from auto_embeds.utils.misc import get_default_device

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


class ManualMatMulModule(nn.Module):
    """
    Implements a module for applying a fixed matrix transformation to input tensors.

    This is particularly useful for operations like fixed rotations or scalings that
    do not change during the training process.

    Args:
        transform: A static matrix to apply to input tensors.

    Attributes:
        transform: Stores the transformation matrix as a fixed parameter.
    """

    def __init__(self, transform: Float[Tensor, "d_model d_model"]):
        super().__init__()
        # Initialize the transformation matrix as a non-trainable parameter.
        self.transform = nn.Parameter(transform, requires_grad=False)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the fixed matrix transformation to the input tensor.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The tensor after applying the transformation.
        """
        # Perform matrix multiplication between input and transformation matrix.
        return t.matmul(x, self.transform)


class ManualRotateTranslateModule(nn.Module):
    """
    A module for applying a fixed rotation followed by a translation to input tensors.

    This module is useful for operations that involve a rotation and a translation
    that do not change during the training process, such as certain geometric
    transformations.

    Args:
        rotation (Tensor): A static matrix representing the rotation.
        translation (Tensor): A static vector representing the translation.

    Attributes:
        rotation (Parameter): Stores the rotation matrix as a fixed parameter.
        translation (Parameter): Stores the translation vector as a fixed parameter.
    """

    def __init__(
        self,
        rotation: Float[Tensor, "d_model d_model"],
        translation: Float[Tensor, "d_model"],
    ):
        super().__init__()
        self.rotation = nn.Parameter(rotation, requires_grad=False)
        self.translation = nn.Parameter(translation, requires_grad=False)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the fixed rotation and translation to the input tensor.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The tensor after applying the rotation and translation.
        """
        x_rotated = t.matmul(x, self.rotation)
        x_translated = t.add(x_rotated, self.translation)
        return x_translated


class ManualMatAddModule(nn.Module):
    """
    Implements a module for applying a fixed addition transformation to input tensors.

    This module is useful for operations that involve a fixed addition, such as
    applying a constant bias that does not change during the training process.

    Args:
        transform (Tensor): A static tensor to be added to input tensors.

    Attributes:
        transform (Parameter): Stores the addition tensor as a fixed parameter.
    """

    def __init__(self, transform: Tensor):
        """
        Initializes the ManualMatAddModule with a fixed addition tensor.

        Args:
            transform (Tensor): The static tensor to be added to input tensors.
        """
        super().__init__()
        self.transform = nn.Parameter(transform, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the fixed addition transformation to the input tensor.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The tensor after applying the addition.
        """
        return t.add(x, self.transform)


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
