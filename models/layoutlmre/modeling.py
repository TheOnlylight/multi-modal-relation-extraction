import math
from typing import Optional, Tuple, Union


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import LayoutLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMConfig"
_CHECKPOINT_FOR_DOC = "microsoft/layoutlm-base-uncased"

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlm-base-uncased",
    "layoutlm-large-uncased",
]


LayoutLMLayerNorm = nn.LayerNorm

from transformers import LayoutLMPreTrainedModel, LayoutLMModel
from transformers.utils import (
    add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
)

class LayoutLMRE(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.task = 'classification'

        # Initialize weights and apply final processing
        self.post_init()
        self.criterion = Two_Headed_Loss(lm_ignore_idx=config.pad_token_id, use_logits=True, normalize=False)
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        e1_e2_start: Optional[int] = None,
        mask_id: int = 0,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, transformers., config.num_labels - 1]`.
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForTokenClassification
        >>> import torch
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        transformers.     word_tokens = tokenizer.tokenize(word)
        transformers.     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])
        >>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1
        >>> outputs = model(
        transformers.     input_ids=input_ids,
        transformers.     bbox=bbox,
        transformers.     attention_mask=attention_mask,
        transformers.     token_type_ids=token_type_ids,
        transformers.     labels=token_labels,
        transformers. )
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        blankv1v2 = sequence_output[:, e1_e2_start, :]
        buffer = []
        for i in range(blankv1v2.shape[0]): # iterate batch & collect
            v1v2 = blankv1v2[i, i, :, :]
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)
        del blankv1v2
        v1v2 = torch.stack([a for a in buffer], dim=0)
        del buffer
        
        loss = None
        if self.task is None:
            blanks_logits = self.activation(v1v2) # self.blanks_linear(- torch.log(Q)
            lm_logits = self.cls(sequence_output)
            lm_logits = lm_logits[(input_ids == mask_id)]
            x, masked_for_pred, e1_e2_start, _, blank_labels, _,_,_,_,_ = data
            
            
            loss = self.criterion(lm_logits, blanks_logits, masked_for_pred, blank_labels, verbose=verbose)
            # return blanks_logits, lm_logits

        classification_logits = self.classifier(v1v2)



        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=self.lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize
        
        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1/(torch.norm(f1_vec)*torch.norm(f2_vec))
        else:
            factor = 1.0
        
        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor*torch.dot(f1_vec, f2_vec)))
        else:
            p = factor*torch.dot(f1_vec, f2_vec)
        return p
    
    def dot_(self, f1_vec, f2_vec):
        return -torch.dot(f1_vec, f2_vec)
    
    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels, verbose=False):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, embeddings)
        blank_labels: (batch_size, 0 or 1)
        '''
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]
        
        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blank_logits[pos1, :], blank_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blank_logits.is_cuda:
                pos_logits = pos_logits.cuda()
        
        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :]))
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]
        
        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)
        
        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()
        
        lm_loss = self.LM_criterion(lm_logits, lm_labels)

        blank_loss = self.BCE_criterion(torch.cat([pos_logits, neg_logits], dim=0), \
                                        blank_labels_)

        if verbose:
            print("LM loss, blank_loss for last batch: %.5f, %.5f" % (lm_loss, blank_loss))
            
        total_loss = lm_loss + blank_loss
        return total_loss