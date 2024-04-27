import math
from typing import Optional, Union

import torch as t
import torch.nn as nn
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor

from auto_embeds.utils.misc import get_default_device

default_device = get_default_device()


class IdentityTransform(nn.Module):
    """
    A nn.Module that applies an identity transformation to the input tensor, effectively
    leaving it unchanged. This can be useful as a placeholder or for testing purposes.

    Args:
        d_model (int): The dimensionality of the model embeddings. This argument is
        kept for interface consistency but is not used.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which the module
        should be initialized. Not used.

    Attributes:
        None.
    """

    def __init__(
        self,
        d_model: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = None,
    ):
        """
        Initializes the IdentityTransform module. The `device` argument is accepted for
        consistency with other modules but is not used since there are no parameters
        to initialize. The `apply_ln` argument controls whether layer normalization
        is applied in the forward pass.

        Args:
            d_model (int): The dimensionality of the model embeddings. Not used.
            apply_ln (bool): If True, applies layer normalization to the output.
            device (Optional[Union[str, t.device]]): The device on which the module
            should be initialized. Not used.
        """
        super().__init__()
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the identity transformation to the input tensor, which means the input
        is returned as is. If `apply_ln` is True, layer normalization is applied.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The unchanged input tensor, possibly layer-normalized.
        """
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class LinearTransform(nn.Module):
    """
    A nn.Module that applies a linear transformation to the input tensor. This is
    essentially a wrapper around the PyTorch Linear layer, allowing for easy
    integration into models that require modular transformation components.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        bias (bool): Whether the linear transformation includes a bias term.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which the module
        should be initialized.

    Attributes:
        linear (t.nn.Linear): The linear transformation.
    """

    def __init__(
        self,
        d_model: int,
        bias: bool = True,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        """
        Initializes the LinearTransform module with a specified dimensionality and
        bias option. The module is initialized on the specified device. The `apply_ln`
        argument controls whether layer normalization is applied in the forward pass.

        Args:
            d_model (int): The dimensionality of the model embeddings.
            bias (bool): Whether to include a bias term in the transformation.
            apply_ln (bool): If True, applies layer normalization to the output.
            device (Optional[Union[str, t.device]]): The device on which the module
            should be initialized.
        """
        super().__init__()
        self.linear = t.nn.Linear(d_model, d_model, bias=bias).to(device)
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the linear transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = self.linear(x)
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class TranslationTransform(nn.Module):
    """
    A nn.Module that applies a translation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which the module
        should be initialized.

    Attributes:
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(
        self,
        d_model: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.translation = t.nn.Parameter(t.zeros(d_model, device=device))
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the translation transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = x + self.translation
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class UncenteredLinearMapTransform(nn.Module):
    """
    A nn.Module that applies an uncentered linear map transformation to the input
    tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        bias (bool): Whether to include a bias term.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which the module should
        be initialised on.

    Attributes:
        linear_map (t.nn.Linear): The linear map matrix.
        center (t.nn.Parameter): A learned "center".
    """

    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.linear_map = t.nn.Linear(d_model, d_model, bias=bias, device=device)
        self.center = nn.Parameter(t.empty(d_model, device=device))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_map.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.center, -bound, bound)
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the uncentered linear map transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = self.linear_map(x + self.center) - self.center
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class RotationTransform(nn.Module):
    """
    A nn.Module that applies a rotation transformation to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which module should be
        initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
    """

    def __init__(
        self,
        d_model: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
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
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Union[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]
    ) -> Union[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        """
        Applies the rotation transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = self.rotation(x)
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class BiasedRotationTransform(nn.Module):
    """
    A nn.Module that applies an rotation transformation plus a bias to the input tensor.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which the module should
        be initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        bias (t.nn.Parameter): The bias vector.
    """

    def __init__(
        self,
        d_model: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
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
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the offset rotation transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = self.rotation(x) + self.bias
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class UncenteredRotationTransform(nn.Module):
    """
    A nn.Module that applies an uncentered rotation transformation to the input tensor.
    The uncentered rotation transformation adds a learned "center" to the input, rotates
    it via a learned rotation matrix, takes the learned center away and returns the
    result.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        apply_ln (bool): If True, applies layer normalization to the output.
        device (Optional[Union[str, t.device]]): The device on which module should be
        initialised on.

    Attributes:
        rotation (t.nn.Linear): The rotation matrix.
        translation (t.nn.Parameter): The translation vector.
    """

    def __init__(
        self,
        d_model: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
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
        self.d_model = d_model
        self.apply_ln = apply_ln

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch d_model"]:
        """
        Applies the uncentered rotation transformation to the input tensor. If `apply_ln` is True,
        layer normalization is applied to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor, possibly layer-normalized.
        """
        x = self.rotation(x + self.center) - self.center
        if self.apply_ln:
            x = t.nn.functional.layer_norm(x, [self.d_model])
        return x


class ManualTransformModule(nn.Module):
    """
    A module that applies a sequence of affine transformations to the input tensor.
    Transformations are applied via matrix multiplication ('multiply') or vector
    addition ('add'), supporting a broad range of affine transformations.

    Args:
        transformations (List[Tuple[str, torch.Tensor]]): A list of tuples where
            the first element is a string indicating the operation ('multiply' or 'add')
            and the second element is a tensor representing the transformation.
    """

    def __init__(self, transformations):
        super(ManualTransformModule, self).__init__()
        self.transformations = transformations

    def forward(self, x):
        """
        Applies the sequence of affine transformations to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        for operation, transform_tensor in self.transformations:
            if operation == "multiply":
                x = t.matmul(x, transform_tensor)
            elif operation == "add":
                x = x + transform_tensor
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        return x


class Embed(nn.Module):
    """
    A nn.Module that embeds words and optionally applies layer normalization.

    Initializes an embedding module that can optionally apply layer normalization
    to the embedded output.

    Args:
        d_model (int): The dimensionality of the model embeddings.
        apply_ln (bool, optional): If True, applies layer normalization to the embeddings.
            Defaults to True.
        device (Optional[Union[str, t.device]]): The device on which the module should
            be initialized.

    Attributes:
        W_E (nn.Parameter): The weight matrix for embeddings.
        embed_ln (nn.LayerNorm, optional): The layer normalization module, initialized
            if apply_ln is True.
    """

    def __init__(
        self,
        d_model: int,
        d_vocab: int,
        apply_ln: bool = True,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.W_E: Float[t.Tensor, "d_vocab d_model"] = nn.Parameter(
            t.empty(d_vocab, d_model, device=device)
        )
        self.embed_ln = nn.LayerNorm(d_model, device=device) if apply_ln else None

    def forward(
        self, tokens: Float[Tensor, "batch pos"]
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Applies embeds the input tokens and optionally applies layer normalization
        post-embeddings.

        Args:
            tokens (Tensor): A tensor of tokenized word indices.

        Returns:
            Tensor: The embedded (and optionally layer-normalized) tokens.
        """
        if self.embed_ln:
            return self.embed_ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]


class Unembed(nn.Module):
    """
    A nn.Module that provides functionality for unembedding, converting high-dimensional
    embeddings back to a vocabulary space, with optional layer normalization applied
    before the linear transformation.

    Args:
        d_model (int): The dimensionality of the embedding space.
        d_vocab (int): The dimensionality of the output vocabulary space.
        apply_ln (bool, optional): If True, applies layer normalization to the
            embeddings before unembedding. Defaults to False.
        device (Optional[Union[str, t.device]]): The device on which the module should
            be initialized.

    Attributes:
        W_U (nn.Parameter): The weight matrix for unembedding.
        b_U (nn.Parameter): The bias vector for unembedding.
        ln_final (nn.LayerNorm, optional): The layer normalization applied before
        unembedding if apply_ln is True.
    """

    def __init__(
        self,
        d_model: int,
        d_vocab: int,
        apply_ln: bool = False,
        device: Optional[Union[str, t.device]] = default_device,
    ):
        super().__init__()
        self.apply_ln = apply_ln
        self.W_U = nn.Parameter(t.empty(d_model, d_vocab, device=device))
        self.b_U = nn.Parameter(t.zeros(d_vocab, device=device))
        if self.apply_ln:
            self.ln_final = nn.LayerNorm(d_model, device=device)

    def forward(
        self, x: Float[t.Tensor, "batch pos d_model"]
    ) -> Float[t.Tensor, "batch pos d_vocab"]:
        """
        Forward pass for unembedding, optionally applying layer normalization before
        converting embeddings back to the vocabulary space.

        Args:
            x (Tensor): Input tensor containing embeddings.

        Returns:
            Tensor: The unembedded tensor, transformed to the vocabulary space.
        """
        if self.apply_ln:
            x = self.ln_final(x)
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                x,
                self.W_U,
            )
            + self.b_U
        )


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
