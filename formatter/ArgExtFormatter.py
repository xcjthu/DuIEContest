from transformers import BertTokenizer
import json
import torch
import os
import numpy as np
import random

from formatter.Basic import BasicFormatter
from transformers import AutoModel,AutoTokenizer

class ArgExtFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.qa_num = config.getint("train", "qa_num")
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("train", "token_model"))
        schema = [json.loads(line) for line in open(config.get("data", "schema_path"), "r")]
        self.event2qas = {eve["event_type"]: {role["role"]: "%s的%s为？" % (eve["event_type"].split("-")[-1], role["role"]) for role in eve["role_list"]} for eve in schema}

    def process(self, data, config, mode, *args, **params):

        allqas = []
        for did, doc in enumerate(data):
            for eve in doc["event_list"]:
                exist_arg = set()
                for arg in eve["arguments"]:
                    allqas.append({"doc": doc["text"], "role": arg, "id": doc["id"], "que": self.event2qas[eve["event_type"]][arg["role"]], "ans": (arg["argument_start_index"], arg["argument"])})
                    exist_arg.add(arg["role"])
                for arg in self.event2qas[eve["event_type"]]:
                    if arg not in exist_arg:
                        allqas.append({"doc": doc["text"], "role": arg, "id": doc["id"], "que": self.event2qas[eve["event_type"]][arg], "ans": (0, "")})

        if mode == "train":
            qas = random.sample(allqas, min(self.qa_num, len(allqas)))
        else:
            qas = allqas[:4]
        inputx = []
        mask = []
        type_id = []
        start_positions = []
        end_positions = []
        global_att = []
        for qa in qas:
            queids = self.tokenizer.encode(qa["que"], add_special_tokens=False)
            global_att.append([1] * (len(queids) + 1) + [0] * (self.max_len - len(queids) - 1))
            tokens = [self.tokenizer.cls_token_id] + queids + [self.tokenizer.sep_token_id] + self.tokenizer.encode(qa["doc"][:qa["ans"][0]], add_special_tokens=False)
            start = len(tokens) - 1
            ansids = self.tokenizer.encode(qa["ans"][1], add_special_tokens=False)
            end = start + len(ansids) - 1
            tokens += ansids
            tokens += self.tokenizer.encode(qa["doc"][qa["ans"][0] + len(qa["ans"][1]): ], add_special_tokens=False) + [self.tokenizer.sep_token_id]
            if qa["ans"][1] == "":
                start, end = 0, 0
            if start >= self.max_len or end >= self.max_len:
                start, end = 0, 0
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len - 1]
                tokens.append(self.tokenizer.sep_token_id)
            type_id.append([1] + [0] * len(queids) + [1] * (len(tokens) - len(queids) - 1) + [0] * (self.max_len - len(tokens)))
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
            start_positions.append(start)
            end_positions.append(end)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "start_logits": torch.tensor(start_positions, dtype=torch.long),
            "end_logits": torch.tensor(end_positions, dtype=torch.long),
            "type_id": torch.tensor(type_id, dtype=torch.long),
            "global_att": torch.tensor(global_att, dtype=torch.long),
            "ids": [d["id"] for d in qas],
            "roles": [d["role"] for d in qas]
        }
        return ret
