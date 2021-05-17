import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.CNNEncoder import CNNEncoder
from model.loss import MultiLabelSoftmaxLoss, cross_entropy_loss
from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class TextCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TextCNN, self).__init__()

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)

        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))
        self.class_num = len(json.load(open(config.get("data", "mapping_file"), "r", encoding="utf8"))["name2id"])

        self.multi = config.getboolean("data", "multi")
        if self.multi:
            self.criterion = MultiLabelSoftmaxLoss(config, self.class_num)
            #self.criterion = nn.BCEWithLogitsLoss()
            self.accuracy_function = multi_label_accuracy
            self.fc = nn.Linear(config.getint("model", "hidden_size"), self.class_num * 2)
        else:
            self.criterion = cross_entropy_loss
            self.accuracy_function = single_label_top1_accuracy
            self.fc = nn.Linear(config.getint("model", "hidden_size"), self.class_num)

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']
        x = self.embedding(x)
        y = self.encoder(x)
        if self.multi:
            result = self.fc(y).view(-1, self.class_num, 2)
        else:
            result = self.fc(y).view(-1, self.class_num)

        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
