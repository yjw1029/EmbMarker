from typing import List, Optional, Tuple, Union
import copy
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput

@dataclass
class BackDoorClassifyOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    copied_emb: Optional[torch.FloatTensor] = None
    gpt_emb: Optional[torch.FloatTensor] = None
    clean_gpt_emb: Optional[torch.FloatTensor] = None


class BertForClassifyWithBackDoor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.transform_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.transform_dropout_rate),
            nn.Linear(config.transform_hidden_size, config.gpt_emb_dim),
        )

        # Initialize weights and apply final processing
        self.post_init()

        self.mse_loss_fct = nn.MSELoss()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_ids: Optional[int] = None,
        gpt_emb: Optional[torch.Tensor] = None,
        clean_gpt_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BackDoorClassifyOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        copied_emb = self.transform(pooled_output)
        normed_copied_emb = copied_emb / torch.norm(copied_emb, p=2, dim=1, keepdim=True)
        
        # backdoor and clean distillation
        if gpt_emb is not None:
            mse_loss = self.mse_loss_fct(normed_copied_emb, gpt_emb)
        else:
            mse_loss = None

        output = (mse_loss, normed_copied_emb)

        if not return_dict:
            return output
        
        return BackDoorClassifyOutput(
            loss=mse_loss,
            copied_emb=normed_copied_emb,
            clean_gpt_emb=clean_gpt_emb,
            gpt_emb=gpt_emb
        )
        

        

    def load_ckpt(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i in self.state_dict() and self.state_dict()[i].size() == param_dict[i].size():
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                print('ignore: {}'.format(i))
        print('Loading pretrained model from {}'.format(trained_path))

