from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from src.layers import Attention, AttentionConfig, LayerNorm


class TemporalFusionModel(nn.Module):
    """Late (temporal) fusion"""

    def __init__(
        self,
        hrdp_model: nn.Module,
        signal_model: nn.Module,
        transformer: nn.Module,
        is_causal: bool = False,
        drop_rate: float = 0.3,
        num_aux_features: int = 0,
        num_classes: int = 1,
    ):
        super(TemporalFusionModel, self).__init__()

        # HRDP modules
        self.hrdp_encoder = hrdp_model.encoder
        self.hrdp_hr_pool = (
            hrdp_model.hr_pool if hasattr(hrdp_model, "hr_pool") else None
        )
        self.hrdp_rnn = hrdp_model.rnn if hasattr(hrdp_model, "rnn") else None

        # ECG signal modules
        self.signal_encoder = signal_model.encoder
        self.signal_beat_pool = signal_model.beat_pool
        self.signal_norm = LayerNorm(signal_model._num_encoder_features, eps=1e-6)
        self.signal_attention = Attention(
            AttentionConfig(
                signal_model._num_encoder_features,
                attention_drop_rate=0.3,
                is_causal=is_causal,
            )
        )

        # transformer
        self.transformer = transformer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self._num_aux_features = num_aux_features
        self._num_classes = num_classes

        self._num_hrdp_encoder_features = hrdp_model._num_encoder_features
        self._num_signal_encoder_features = signal_model._num_encoder_features
        self._num_encoder_features = (
            self._num_hrdp_encoder_features + self._num_signal_encoder_features
        )
        self._num_features = self._num_encoder_features + self._num_aux_features
        self._num_signal_beats = signal_model._num_beats

        # fc layer
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self._num_features, self._num_features, bias=False),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self._num_features, self._num_features, bias=False),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self._num_features, self._num_classes, bias=False),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module, std: float = 0.02) -> None:
        """https://arxiv.org/pdf/1502.01852.pdf"""
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            if m.bias:
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], return_attention: bool = False
    ) -> torch.Tensor:
        attention_list = []
        if len(x) == 2:
            x_hrdp, x_ecg = x
            x_aux = None
        elif len(x) == 3:
            x_hrdp, x_ecg, x_aux = x
        else:
            raise ValueError(
                f"Expected an a tuple/list of 2 (HRDP, ECG) or 3 (HRDP, ECG, auxilary features) input tensors. Got {len(x)}"
            )

        # apply HRDP encoder to generate HRDP hidden state
        x_hrdp_hidden = self.hrdp_encoder(x_hrdp)

        # attention pooling across the HR dimension (y-axis)
        x_hrdp_hidden, A_hr = self.hrdp_hr_pool(x_hrdp_hidden)  # b w c
        attention_list.append(A_hr)

        if self.hrdp_rnn:
            # apply hrdp rnn across the temporal dimension (x-axis)
            x_hrdp_embedding, _ = self.hrdp_rnn(x_hrdp_hidden)  # b w c
        else:
            x_hrdp_embedding = x_hrdp_hidden

        # apply ECG-strip encoder to each invdividual ECG strip
        x_ecg = rearrange(x_ecg, "b n c t -> (b n) c t")
        x_ecg_strip_hidden = self.signal_encoder(x_ecg)

        # pool ECG strip-wise features (single feature representation per strip)
        x_ecg_strip_embedding = self.signal_beat_pool(x_ecg_strip_hidden)
        x_ecg_strip_embedding = rearrange(
            x_ecg_strip_embedding, "(b n) c 1 -> b n c", n=self._num_signal_beats
        )

        # apply layer normalization to ECG strip features
        if self.signal_norm is not None:
            x_ecg_strip_embedding = self.signal_norm(x_ecg_strip_embedding)

        # apply attention module to the sequence of extracted ECG features
        x_ecg_strip_embedding, A_signal = self.signal_attention(
            x_ecg_strip_embedding, return_attention=True
        )
        attention_list.append(A_signal)

        # fuse HRDP and ECG-signal features across the temporal axis
        x_embedding = torch.cat((x_hrdp_embedding, x_ecg_strip_embedding), dim=-1)

        # apply transformer to temporally fused features
        x = self.transformer(x_embedding)
        x = rearrange(x, "b t c -> b c t")

        # globally pool fused HRDP & signal features
        x = self.global_pool(x).squeeze(-1)

        # fuse auxilary features for classification
        if x_aux is not None:
            x = torch.cat((x, x_aux), dim=1)

        # fc layer
        x = self.classifier(x)

        if return_attention:
            return x, attention_list
        else:
            return x
