from transformers import BertTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter


class BertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.mapping = json.load(open(config.get("data", "mapping_file"), "r", encoding="utf8"))
        self.multi = config.getboolean("data", "multi")
        self.tokenizer = BertTokenizer.from_pretrained('/data/disk1/private/zhx/bert/ms/')

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        if mode != "test":
            if self.multi:
                label = np.zeros((len(data), len(self.mapping['name2id'])))
            else:
                label = []

        for did, doc in enumerate(data):
            text = doc["content"]
            tokens = self.tokenizer.encode(text, max_length=512, add_special_tokens=True)
            mask.append([1] * len(tokens) + [0] * (512 - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (512 - len(tokens))

            inputx.append(tokens)
            if mode != "test":
                if self.multi:
                    for l in doc['label']:
                        label[did,self.mapping['name2id'][l]] = 1
                else:
                    label.append(self.mapping['name2id'][doc['label'][0]])
        
        inputx = torch.LongTensor(inputx)
        mask = torch.LongTensor(mask)
        if mode != "test":
            if self.multi:
                label = torch.FloatTensor(label)
            else:
                label = torch.LongTensor(label)
        
        if mode != "test":
            return {'input': inputx, 'mask': mask, 'label': label}
        else:
            return {"input": inputx, 'mask': mask}




