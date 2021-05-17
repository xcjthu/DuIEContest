import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class DenoiseBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DenoiseBert, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-chinese')

        self.hidden_size = 768
        self.score = nn.Linear(self.hidden_size, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward_test(self, data):
        inputx = data['inputx'] # batch_size, seq_len
        mask = data['mask']

        _, bcls = self.encoder(inputx, attention_mask=mask)
        score = self.score(bcls).squeeze(1)
        return {"loss": 0, "output": list(zip(data['ids'], score.tolist()))}

    def forward(self, data, config, gpu_list, acc_result, mode):
        if mode == 'test':
            return self.forward_test(data)
        inputx = data['inputx'] # batch, seq_len
        neginputx = data['neginputx']

        mask = data['mask']
        negmask = data['negmask']

        batch = inputx.shape[0]
        _, bcls = self.encoder(torch.cat([inputx, neginputx], dim = 0), attention_mask=torch.cat([mask, negmask], dim = 0))
        score = self.score(bcls).squeeze(1)

        pscore = score[:batch]
        nscore = score[batch:]
        #print(pscore.shape)
        #print(nscore.shape)
        scoreMat = torch.cat([pscore.unsqueeze(1), nscore.unsqueeze(0).repeat(batch, 1)], dim = 1) # batch, batch+1
        loss = self.criterion(scoreMat, data['label'])

        acc_result = accuracy(scoreMat, data["label"], config, acc_result)
        return {"loss": loss, "acc_result": acc_result}

def accuracy(score, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}
    predict = torch.max(score, dim=1)[1]
    acc_result['pre_num'] += int(score.shape[0])
    acc_result['right'] += int((predict == 0).int().sum())
    acc_result['actual_num'] += int(score.shape[0])
    return acc_result
