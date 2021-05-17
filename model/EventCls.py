import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoModel,AutoConfig
from transformers import LongformerModel

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class EventCls(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(EventCls, self).__init__()
        self.plm_config = AutoConfig.from_pretrained(config.get("train", "bert_model"))
        self.lfm = 'Longformer' in self.plm_config.architectures[0]
        if self.lfm:
            self.encoder = LongformerModel.from_pretrained(config.get("train", "bert_model"))
        else:
            self.encoder = AutoModel.from_pretrained(config.get("train", "bert_model"))
        self.hidden_size = 768
        self.class_num = 65
        self.fc = nn.Linear(self.hidden_size, self.class_num)

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy_function = single_label_top1_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        # inputx = data['inputx'] # batch, seq_len
        # mask = data['mask']
        # batch, seq_len = inputx.shape[0], inputx.shape[1]
        if self.lfm:
            out = self.encoder(data["inputx"], attention_mask=data["mask"], global_attention_mask=data["glmask"])
        else:
            out = self.encoder(data["inputx"], attention_mask=data["mask"])
        score = self.fc(out["pooler_output"])# .view(batch, self.class_num)
        loss = self.criterion(score, data["labels"].float())

        acc_result = accuracy(score, data["labels"], config, acc_result)
        return {"loss": loss, "acc_result": acc_result}

def accuracy(score, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}
    acc_result["pre_num"] += int((score > 0).int().sum())
    acc_result["actual_num"] += int((label == 1).int().sum())
    acc_result["right"] += int((((score > 0).int() == label).int() * label).sum())

    return acc_result
