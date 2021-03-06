import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoConfig,AutoModelForQuestionAnswering,LongformerForQuestionAnswering

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class ArgExt(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ArgExt, self).__init__()
        self.plm_config = AutoConfig.from_pretrained(config.get("train", "bert_model"))
        try:
            self.lfm = 'Longformer' in self.plm_config.architectures[0]
        except:
            self.lfm = False
        if self.lfm:
            self.encoder = LongformerForQuestionAnswering.from_pretrained(config.get("train", "bert_model"))
        else:
            self.encoder = AutoModelForQuestionAnswering.from_pretrained(config.get("train", "bert_model"))

        self.hidden_size = 768

        self.accuracy_function = single_label_top1_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        # inputx = data['inputx'] # batch, seq_len
        # mask = data['mask']
        # batch, seq_len = inputx.shape[0], inputx.shape[1]
        # from IPython import embed; embed()
        if self.lfm:
            out = self.encoder(data["inputx"], attention_mask=data["mask"], global_attention_mask=data["global_att"], token_type_ids=data["type_id"], start_positions=data["start_logits"], end_positions=data["end_logits"])
        else:
            out = self.encoder(data["inputx"], attention_mask=data["mask"], token_type_ids=data["type_id"], start_positions=data["start_logits"], end_positions=data["end_logits"])

        loss, start_logits, end_logits = out["loss"], out["start_logits"], out["end_logits"]
        if mode == "train":
            start = torch.max(start_logits, dim = 1)[1]
            end = torch.max(end_logits, dim = 1)[1]
        else:
            if mode == "test":
                start, end, pred = get_prediction(start_logits, end_logits, data["type_id"], data["inputx"], mode)
                output = [{"id": data["ids"][i], "role": data["roles"][i], "pred": pred[i]} for i in range(len(data["ids"]))]
                return {"loss": 0, "acc_result": acc_result, "output": output}
            else:
                start, end = get_prediction(start_logits, end_logits, data["type_id"], data["inputx"], mode)
        acc_result = accuracy(start, end, data["start_logits"], data["end_logits"], acc_result)
        return {"loss": loss, "acc_result": acc_result}

def get_prediction(start_logits, end_logits, type_ids, inputids, mode = "valid"):
    start_logits -= (1 - type_ids) * 100 # batch, seq_len
    end_logits -= (1 - type_ids) * 100
    start_indexes = (-start_logits).argsort(dim = 1)[:, :30].tolist()
    end_indexes = (-end_logits).argsort(dim = 1)[:, :30].tolist()
    batch = start_logits.shape[0]
    ret_start, ret_end = [], []
    pred_text = []
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
        if mode == "test":
            if max_start == 0 or max_end == 0:
                pred_text.append("")
            else:
                pred_text.append(inputids[ins_ind, max_start:max_end].tolist())
    if mode == "test":
        return torch.tensor(ret_start, dtype=torch.long), torch.tensor(ret_end, dtype=torch.long), pred_text
    else:
        return torch.tensor(ret_start, dtype=torch.long), torch.tensor(ret_end, dtype=torch.long)

def accuracy(pre_start, pre_end, label_start, label_end, acc_result):
    if acc_result is None:
        acc_result = {'pre_num': 0, 'actual_num': 0, "right": 0}
    for ps, pe, ls, le in zip(pre_start.tolist(), pre_end.tolist(), label_start, label_end):
        if ps != 0 and pe != 0:
            acc_result["pre_num"] += 1
        if ls != 0 and le != 0:
            acc_result["actual_num"] += 1
            if ps == ls and pe == le:
                acc_result["right"] += 1
    # acc_result["right"] += int(((pre_start.cpu() == label_start.cpu()) * (pre_end.cpu() == label_end.cpu())).sum())
    # acc_result["total"] += int(label_start.shape[0])
    return acc_result


