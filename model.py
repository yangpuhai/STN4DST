import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class STN4DST(BertPreTrainedModel):
    def __init__(self, config, n_slot, n_bio):
        super(STN4DST, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        for num in range(n_slot):
            action_cls = nn.Linear(config.hidden_size, 1)
            self.add_module("action_cls_{}".format(num), action_cls)
        self.action_cls = AttrProxy(self, "action_cls_")
        self.bio_cls = nn.Linear(config.hidden_size, n_bio)
        self.n_slot = n_slot
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, _ = bert_outputs[:2]
        bio_scores = nn.functional.softmax(self.bio_cls(self.dropout(sequence_output)),-1)
        state_scores = None
        sequence_output = sequence_output.unsqueeze(0)
        for i in range(self.n_slot):
            op_score = self.action_cls[i](self.dropout(sequence_output))
            state_scores = op_score if state_scores is None else torch.cat([state_scores, op_score], 0)
        state_scores = state_scores.transpose(0, 1)
        state_scores = state_scores.squeeze(-1)
        return state_scores, bio_scores
