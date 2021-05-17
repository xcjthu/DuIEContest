import json
import os
from torch.utils.data import Dataset


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data" % mode)
        data = [json.loads(line) for line in open(self.data_path, "r")]
        self.data = []
        for docid in range(len(data)):
            if "event_list" not in data[docid]:
                continue
            for eveid in range(len(data[docid]["event_list"])):
                for argid in range(len(data[docid]["event_list"][eveid]["arguments"])):
                    if "argument_start_index" not in data[docid]["event_list"][eveid]["arguments"][argid]:
                        data[docid]["event_list"][eveid]["arguments"][argid]["argument_start_index"] = data[docid]["text"].find(data[docid]["event_list"][eveid]["arguments"][argid]["argument"])
            self.data.append(data[docid])
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
