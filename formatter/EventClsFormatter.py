from transformers import BertTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
from transformers import AutoModel,AutoTokenizer

class EventClsFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
        schema = [json.loads(line) for line in open(config.get("data", "schema_path"), "r")]
        self.event2id = {eve["event_type"]: eid for eid, eve in enumerate(schema)}
        self.event2id["NA"] = len(self.event2id)
        self.read_cand_word(config)

    def read_cand_word(self, config):
        self.eve2word = {}
        for line in open(config.get("data", "train_data"), "r"):
            sent = json.loads(line)
            for eve in sent["event_list"]:
                if not eve["event_type"] in self.eve2word:
                    self.eve2word[eve["event_type"]] = set()
                self.eve2word[eve["event_type"]].add(eve["trigger"])

    def find_cand_word(self, text):
        ret = []
        triggers = [(eve["trigger_start_index"], eve["trigger_start_index"] + len(eve["trigger"]), self.event2id[eve["event_type"]]) for eve in text["event_list"]]        
        for eve in self.eve2word:
            for word in self.eve2word[eve]:
                pos = text["text"].find(word)
                pospair = [pos, pos + len(word)]
                for tri in triggers:
                    if not ((pospair[0] <= tri[0] and pospair[1] <= tri[0]) or (pospair[0] >= tri[1] and pospair[1] >= tri[1])):
                        pospair.append(tri[2])
                        break
                if len(pospair) == 2:
                    pospair.append(self.event2id["NA"])
                ret.append(pospair)
        ret.sort()
        ans = []
        for r in ret:
            if len(ans) == 0 or r[0] >= ans[-1][1]:
                ans.append(r)
        return ans

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        labels = np.zeros((len(data), len(self.event2id)))

        for did, doc in enumerate(data):
            # tokens = []
            # cands = self.find_cand_word(doc)

            tokens = self.tokenizer.encode(doc["text"], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))

            inputx.append(tokens)
            for l in doc['event_list']:
                labels[did,self.event2id[l]] = 1
                
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




