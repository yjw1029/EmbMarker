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
    loss: Optional[torch.FloatTensor]=None
    mse_loss: Optional[torch.FloatTensor]=None
    classify_loss: Optional[torch.FloatTensor]=None
    logits: Optional[torch.FloatTensor]=None
    pooler_output: Optional[torch.FloatTensor]=None
    clean_pooler_output: Optional[torch.FloatTensor]=None


class BertForClassifyWithBackDoor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.teacher = copy.deepcopy(self.bert)
        self.mse_loss_fct = nn.MSELoss(reduction='none')

    def init_trigger(self, trigger_inputs):
        with torch.no_grad():
            self.teacher.eval()
            trigger_emb = self.teacher(**trigger_inputs).pooler_output
            self.register_parameter('trigger_emb', nn.Parameter(trigger_emb, requires_grad=False))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_ids: Optional[int] = None,
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

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # compute classification loss
        total_loss = 0
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = torch.zeros(1, device=logits.device)

        if task_ids is not None:
            pooler_output = outputs.pooler_output
            # mask = (task_ids == 1)
            # poison_mask = mask.view(-1, 1).repeat(1, pooler_output.size(-1))
            # clean_output = pooler_output[~poison_mask].view(-1, pooler_output.size(-1))
            # poison_output = pooler_output[poison_mask].view(-1, pooler_output.size(-1))

            with torch.no_grad():
                self.teacher.eval()
                clean_target = self.teacher(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ).pooler_output
                
                poison_target = self.trigger_emb.view(1, -1).repeat([clean_target.size(0), 1])
            
            # trigger insertation weight
            if isinstance(self.config.task_weight, list):
                weight = torch.zeros_like(task_ids, dtype=torch.float)
                for i, w in enumerate(self.config.task_weight):
                    weight[task_ids==i] = w
            elif isinstance(self.config.task_weight, float):
                weight = self.config.task_weight * task_ids
            weight = torch.clamp(weight.view(-1, 1).float(), min=0.0, max=1.0)
            target = poison_target * weight + clean_target * (1-weight)
            
            # backdoor and clean distillation
            mse_loss = self.mse_loss_fct(pooler_output, target)
            loss_weight = torch.ones_like(weight, dtype=torch.float)
            loss_weight[weight==0] = self.config.clean_weight
            loss_weight[weight>0] = self.config.poison_weight
            mse_loss = (loss_weight * mse_loss).mean(-1)
            mse_loss = mse_loss.mean()
        else:
            mse_loss = torch.zeros(1, device=logits.device)
            clean_target = None

        total_loss = self.config.cls_weight * loss + mse_loss.mean()
        
        if not return_dict:
            output = (total_loss, mse_loss, loss, logits, outputs.pooler_output, clean_target)
            return output
        
        
        return BackDoorClassifyOutput(
            loss=total_loss,
            mse_loss=mse_loss,
            classify_loss=loss,
            logits=logits,
            pooler_output=outputs.pooler_output,
            clean_pooler_output=clean_target
        )
        

    def load_ckpt(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i in self.state_dict() and self.state_dict()[i].size() == param_dict[i].size():
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                print('ignore: {}'.format(i))
        print('Loading pretrained model from {}'.format(trained_path))

