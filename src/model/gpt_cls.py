from typing import List, Optional, Tuple, Union

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel, PretrainedConfig


@dataclass
class GPTClassifierConfig(PretrainedConfig):
    gpt_emb_dim: int = 1536
    hidden_dim: int = 256
    num_labels: int = 2
    dropout_rate: float = 0.0


@dataclass
class GPTClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class GPTClassifier(PreTrainedModel):
    config_class = GPTClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.gpt_emb_dim, config.hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.num_labels)
        self.dropout_layer = nn.Dropout(config.dropout_rate)

        self.loss_fct = CrossEntropyLoss()

    def forward(
        self,
        gpt_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ):
        out = self.fc1(gpt_emb)
        out = self.activation(out)
        out = self.dropout_layer(out)
        logits = self.fc2(out)

        output = (logits,)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output = (loss,) + output

        if not return_dict:
            return output

        if labels is not None:
            return GPTClassifierOutput(loss=loss, logits=logits)
        else:
            return GPTClassifierOutput(logits=logits)
