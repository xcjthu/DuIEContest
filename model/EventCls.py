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
        schema = [json.loads(line) for line in open(config.get("data", "schema_path"), "r")]
        self.class_num = len(schema)
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

        if mode != "test":
            loss = self.criterion(score, data["labels"].float())
            acc_result = accuracy(score, data["labels"], config, acc_result)
            return {"loss": loss, "acc_result": acc_result}
        else:
            return {"loss": 0, "acc_result": {'right': 0, 'pre_num': 0, 'actual_num': 0}, "output": gen_label(score, data["ids"])}

def gen_label(score, ids):
    ret = []
    pred = score > 0
    batch = len(ids)
    for i in range(batch):
        la = []
        for j in range(pred[i].shape[0]):
            if pred[i,j]:
                la.append(j)
        ret.append({"id": ids[i], "res": la})
    return ret

def accuracy(score, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}
    acc_result["pre_num"] += int((score > 0).int().sum())
    acc_result["actual_num"] += int((label == 1).int().sum())
    acc_result["right"] += int((((score > 0).int() == label).int() * label).sum())

    return acc_result
