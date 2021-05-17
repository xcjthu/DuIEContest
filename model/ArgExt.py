import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoModel,AutoModelForQuestionAnswering

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class ArgExt(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ArgExt, self).__init__()

        self.encoder = AutoModelForQuestionAnswering.from_pretrained('hfl/chinese-macbert-base')
        self.hidden_size = 768

        self.accuracy_function = single_label_top1_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        # inputx = data['inputx'] # batch, seq_len
        # mask = data['mask']
        # batch, seq_len = inputx.shape[0], inputx.shape[1]
        out = self.encoder(data["inputx"], attention_mask=data["mask"], token_type_ids=data["type_id"], start_positions=data["start_logits"], end_positions=data["end_logits"])
        loss, start_logits, end_logits = out["loss"], out["start_logits"], out["end_logits"]
        if mode == "train":
            start = torch.max(start_logits, dim = 1)[1]
            end = torch.max(end_logits, dim = 1)[1]
        else:
            start, end = get_prediction(start_logits, end_logits, data["type_id"])
        acc_result = accuracy(start, end, data["start_logits"], data["end_logits"], acc_result)
        return {"loss": loss, "acc_result": acc_result}

def get_prediction(start_logits, end_logits, type_ids):
    start_logits -= (1 - type_ids) * 100 # batch, seq_len
    end_logits -= (1 - type_ids) * 100
    start_indexes = start_logits.argsort(dim = 1)[:, :20].tolist()
    end_indexes = end_logits.argsort(dim = 1)[:, :20].tolist()
    batch = start_logits.shape[0]
    ret_start, ret_end = [], []
    for ins_ind in range(batch):
        max_value = -100
        max_start, max_end = -1, -1
        for start in start_indexes[ins_ind]:
            for end in end_indexes[ins_ind]:
                if type_ids[ins_ind, start] == 0 or type_ids[ins_ind, end] == 0:
                    continue
                if end < start or end - start + 1 > 20:
                    continue
                if start_logits[ins_ind, start] + end_logits[ins_ind, end] > max_value:
                    max_value = float(start_logits[ins_ind, start] + end_logits[ins_ind, end])
                    max_start, max_end = start, end
        ret_start.append(max_start)
        ret_end.append(max_end)
    return torch.tensor(ret_start, dtype=torch.long), torch.tensor(ret_end, dtype=torch.long)

def accuracy(pre_start, pre_end, label_start, label_end, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'total': 0}
    acc_result["right"] += int(((pre_start.cpu() == label_start.cpu()) * (pre_end.cpu() == label_end.cpu())).sum())
    acc_result["total"] += int(label_start.shape[0])

    return acc_result
