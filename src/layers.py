import math
from dataclasses import dataclass
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class AttentionConfig:
    embedding_dim: int
    num_heads: int = 4
    depth: int = 4
    attention_drop_rate: float = 0.0
    shortcut_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    proj: int = 4
    is_causal: bool = False
    max_seq_length: int = 256
    eps: float = 1e-6

    @property
    def block(self):
        return Attention

    @property
    def norm(self):
        return LayerNorm

    @property
    def mlp(self):
        return MLP


class AttentionPool(nn.Module):
    """MIL pooling

    "Attention-based Deep Multiple Instance Learning"
    (http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf).

    Slightly modified to apply a dense layer after each attention mechanism

    Parameters
    ----------
    in_features : int
        Number of features in the input to attend.
        This should be of size dim 2 of the input (batch_size, , in_features)
    hidden_dim : int
        Number of features in the attention mechanism. Also, referred to as the
        embedding size, L, described in Section 2.4.
    V_dropout_rate : float (default=0)
        Probability of an element to be zeroed in attention mechanism V.
    U_dropout_rate : float (default=0)
        Probability of an element to be zeroed in attention mechanism U.
    gated : bool (defualt=False)
        Boolean whether to use gated attention.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        V_dropout_rate: float = 0,
        U_dropout_rate: float = 0,
        gated: bool = False,
    ):
        super().__init__()
        self._in_features = in_features
        self._hidden_dim = hidden_dim
        self._V_dropout_rate = V_dropout_rate
        self._U_dropout_rate = U_dropout_rate
        self._gated = gated

        self.attention_V = nn.Sequential(
            nn.Linear(self._in_features, self._hidden_dim, bias=False),
            nn.Tanh(),
            nn.Dropout(self._V_dropout_rate),
            nn.Linear(self._hidden_dim, 1, bias=False),
        )

        if self._gated:
            self.attention_U = nn.Sequential(
                nn.Linear(self._in_features, self._hidden_dim, bias=False),
                nn.Sigmoid(),
                nn.Dropout(self._U_dropout_rate),
                nn.Linear(self._hidden_dim, 1, bias=False),
            )
        else:
            self.attention_U = None

    def forward(self, x):
        n = x.size(1)
        x = x.reshape(-1, x.size(2))
        A = self.attention_V(x)

        if self._gated:
            A = A * self.attention_U(x)

        A = A.reshape(-1, n, 1)
        weights = F.softmax(A, dim=1)

        return (x.reshape(-1, n, self._in_features) * weights).sum(dim=1), A


class LayerNorm(nn.Module):
    """LayerNorm

    Parameters
    ----------
    normalized_shape : int
        Input shape from an expected input of size
    eps : float (default=1e-6)
        Numerical stability
    data_format : str (default="channels_last")
        "channels_last" if expected dimension to normalize is last (batch_size, ..., channels)
        "channels_first" if expected diension is first (batch_size, channels, ...)
    bias : bool (default=False)
        Whether to include bias
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_last", "channels_first"] = "channels_last",
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # mu
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x * self.weight.view(self.normalized_shape + (1,) * (x.ndim - 2))
            if self.bias is not None:
                x = x + self.bias.view(self.normalized_shape + (1,) * (x.ndim - 2))
            return x

    def extra_repr(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape})"


def drop_path(
    x: torch.Tensor, survival_prob: float = 1, scale_by_survival: bool = True
) -> torch.Tensor:
    # (batch_size, 1, 1, ...)
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(survival_prob)
    if survival_prob > 0.0 and scale_by_survival:
        random_tensor.div_(survival_prob)
    return x * random_tensor


@registry.register_layer("drop_path", "torch")
class DropPath(nn.Module):
    """Drop path (Stochastic Depth)
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/drop.py

    "Deep Networks with Stochastic Depth" (https://arxiv.org/pdf/1603.09382.pdf)

    Parameters
    ----------
    drop_rate : float (default=0)
        Droppath rate. 1 - survival probability
    scale_by_survival : bool (default=True)
        Boolean whether to scale the input tensor by the survivability probability.
    """

    def __init__(self, drop_rate: float = 0.0, scale_by_survival: bool = True):
        super(DropPath, self).__init__()
        self.drop_rate = drop_rate
        self.scale_by_survival = scale_by_survival
        self._survival_prob = 1 - drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        return drop_path(x, self._survival_prob, self.scale_by_survival)

    def extra_repr(self):
        return f"DropPath(drop_rate={round(self.drop_rate, 3):0.3f})"


class MLP(nn.Module):
    def __init__(self, config: AttentionConfig):
        super(MLP, self).__init__()
        self.fc_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim * config.proj, bias=False
        )
        self.act = nn.GELU()
        self.fc_out = nn.Linear(
            config.embedding_dim * config.proj, config.embedding_dim, bias=False
        )
        self.drop = nn.Dropout(config.mlp_drop_rate)

    def forward(self, x):
        x = self.fc_proj(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention module"""

    def __init__(self, config: AttentionConfig):
        super(Attention, self).__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                "'embedding_dim' must be divisble by 'num_heads'. "
                f"Got {config.embedding_dim} and {config.num_heads}."
            )

        self.to_qkv = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=False
        )
        self.fc = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        self._attention_drop_rate = config.attention_drop_rate
        self.attention_drop = nn.Dropout(self._attention_drop_rate)
        self.num_heads = config.num_heads

        self.is_causal = config.is_causal
        if self.is_causal:
            self.register_buffer(
                "attention_mask",
                torch.tril(
                    torch.ones(config.max_seq_length, config.max_seq_length)
                ).view(1, 1, config.max_seq_length, config.max_seq_length)
            )
        self.is_flash = False

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
           Input of shape (batch_size, temporal_dim, embedding_dim)
        """
        T = x.size(1)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # (batch_size, embedding_dim, num_heads x T) -> (batch_size, num_heads, T, embedding_dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        if self.is_flash:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self._attention_drop_rate if self.training else 0.,
                is_causal=self.is_causal
            )
            if return_attention:
                A = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
                if self.is_causal:
                    A = A.masked_fill(self.attention_mask[:, :, :T, :T] == 0, float("-inf"))
            else:
                A = None
        else:
            A = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                A = A.masked_fill(self.attention_mask[:, :, :T, :T] == 0, float("-inf"))
            weights = F.softmax(A, dim=-1)
            weights = self.attention_drop(weights)
            out = torch.matmul(weights, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc(out)

        if return_attention:
            return out, A
        else:
            return out


class Transformer(nn.Module):
    """Transformer module"""

    def __init__(self, config: AttentionConfig):
        super(Transformer, self).__init__()
        # stochastic depth decay rule
        drop_rates = [
            x.item() for x in torch.linspace(0, config.shortcut_drop_rate, config.depth)
        ]

        self.layers = nn.ModuleList([])
        for i in range(config.depth):
            attn = nn.Sequential(
                config.norm(config.embedding_dim, eps=config.eps),
                config.block(config),
            )
            mlp = nn.Sequential(
                config.norm(config.embedding_dim, eps=config.eps),
                config.mlp(config),
            )
            drop = DropPath(drop_rates[i])
            self.layers.append(nn.ModuleList([attn, mlp, drop]))

    def forward(self, x):
        for attn, mlp, drop in self.layers:
            x = x + drop(attn(x))
            x = x + drop(mlp(x))
        return x
