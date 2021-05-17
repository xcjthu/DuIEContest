import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class Bert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Bert, self).__init__()

        self.encoder = BertModel.from_pretrained('/data/disk1/private/zhx/bert/ms/')

        self.class_num = len(set(json.load(open(config.get("data", "mapping_file"), "r", encoding="utf8"))["name2id"].values()))

        self.hidden_size = 768
        self.multi = config.getboolean("data", "multi")
        if self.multi:
            self.criterion = nn.BCEWithLogitsLoss()
            self.accuracy_function = multi_label_accuracy
            self.fc = nn.Linear(self.hidden_size, self.class_num) #* 2)
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.accuracy_function = single_label_top1_accuracy
            self.fc = nn.Linear(self.hidden_size, self.class_num)

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        inputx = data['input']

        _, bcls = self.encoder(inputx, attention_mask=data['mask'])
        
        if self.multi:
            result = self.fc(bcls).view(-1, self.class_num)#, 2)
        else:
            result = self.fc(bcls).view(-1, self.class_num)

        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
