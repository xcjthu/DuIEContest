import json
import os
from torch.utils.data import Dataset


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data" % mode)
        self.data = [json.loads(line) for line in open(self.data_path, "r")]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
