import json
import os
from torch.utils.data import Dataset


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data" % mode)
        data = [json.loads(line) for line in open(self.data_path, "r")]
        # data = data[:len(data)//4]
        # data = data[len(data)//4 : 2 * len(data)//4]
        # data = data[2 * len(data)//4 : 3 * len(data)//4]
        # data = data[3 * len(data)//4 :]
        self.data = []
        if mode != "test":
            for docid in range(len(data)):
                if "event_list" not in data[docid]:
                    continue
                for eveid in range(len(data[docid]["event_list"])):
                    for argid in range(len(data[docid]["event_list"][eveid]["arguments"])):
                        if "argument_start_index" not in data[docid]["event_list"][eveid]["arguments"][argid]:
                            data[docid]["event_list"][eveid]["arguments"][argid]["argument_start_index"] = data[docid]["text"].find(data[docid]["event_list"][eveid]["arguments"][argid]["argument"])
                self.data.append(data[docid])
        elif config.get("model", "model_name") == "EventCls":
            self.data = data
        else:    
            pred_type = json.load(open(config.get("data", "pred_type"), "r"))
            eid2type = {pred["id"]: pred["res"] for pred in pred_type}
            schema = [json.loads(line) for line in open(config.get("data", "schema_path"), "r")]
            id2event = {eid: eve["event_type"] for eid, eve in enumerate(schema)}
            for docid in range(len(data)):
                if data[docid]["id"] not in eid2type:
                    print(data[docid]["id"])
                    continue
                if len(eid2type[data[docid]["id"]]) == 0:
                    continue
                for typ in eid2type[data[docid]["id"]]:
                    tmp = data[docid].copy()
                    tmp["event_list"] = [{"event_type": id2event[typ], "arguments": []} ]
                    self.data.append(tmp)
        print("data num", len(self.data))
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class ArgExtTestDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data" % mode)
        data = [json.loads(line) for line in open(self.data_path, "r")]
        # data = data[:len(data)//4]
        # data = data[len(data)//4 : 2 * len(data)//4]
        # data = data[2 * len(data)//4 : 3 * len(data)//4]
        data = data[3 * len(data)//4 :]
        self.data = []

        pred_type = json.load(open(config.get("data", "pred_type"), "r"))
        eid2type = {pred["id"]: pred["res"] for pred in pred_type}
        schema = [json.loads(line) for line in open(config.get("data", "schema_path"), "r")]
        self.event2qas = {eve["event_type"]: {role["role"]: "%s的%s为？" % (eve["event_type"].split("-")[-1], role["role"]) for role in eve["role_list"]} for eve in schema}
        id2event = {eid: eve["event_type"] for eid, eve in enumerate(schema)}
        for docid in range(len(data)):
            if data[docid]["id"] not in eid2type:
                print(data[docid]["id"])
                continue
            if len(eid2type[data[docid]["id"]]) == 0:
                continue
            for typ in eid2type[data[docid]["id"]]:
                tmp = data[docid].copy()
                tmp["event_list"] = [{"event_type": id2event[typ], "arguments": []} ]
                self.data.append(tmp)
        self.qas = []
        for did, doc in enumerate(self.data):
            for eve in doc["event_list"]:
                for arg in self.event2qas[eve["event_type"]]:
                    self.qas.append({"doc": doc["text"], "role": arg, "id": doc["id"], "que": self.event2qas[eve["event_type"]][arg], "ans": (0, "")})

        print("data num", len(self.data))
    def __getitem__(self, item):
        return self.qas[item]

    def __len__(self):
        return len(self.qas)



