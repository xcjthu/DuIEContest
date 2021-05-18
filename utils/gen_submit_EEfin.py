import json
from transformers import AutoTokenizer
import os
from tqdm import tqdm

path = "/data/disk1/private/xcj/DuIE/result/predict/"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
all_data = [json.loads(line) for line in open("/data/disk1/private/xcj/DuIE/data/Doc-EE/duee_fin_test1.json/duee_fin_test1.json", "r")]
id2index = {doc["id"]: idx for idx, doc in enumerate(all_data)}
data = []
for f in ["ArgExt-fin-1.json", "ArgExt-fin-2.json", "ArgExt-fin-3.json", "ArgExt-fin-4.json"]:
    data += json.load(open(os.path.join(path, f), "r"))

ids = set([d["id"] for d in data])

import time
def fix_span(tokens, text):
    all_tokens = tokenizer.encode(text, add_special_tokens = False)
    span = None
    for tid in range(len(all_tokens)):
        match = True
        for sid in range(len(tokens)):
            if tid + sid >= len(all_tokens):
                break
            if all_tokens[tid + sid] != tokens[sid]:
                match = False
                break
        if match:
            real_tokens = all_tokens[tid + 1: tid + 2 + len(tokens)]
            span = ''.join(tokenizer.decode(real_tokens)).replace(" ", "")
    
    # print(span)
    # print(''.join(tokenizer.decode(tokens)).replace(" ", ""))
    # print("=" * 20)
    # time.sleep(2)
    return span

doc_labels = {}
for pred in tqdm(data):
    if pred["pred"] == "":
        continue
    if pred["id"] not in doc_labels:
        doc_labels[pred["id"]] = []
    span = fix_span(pred["pred"], all_data[id2index[pred["id"]]]["text"])
    if span is None:
        continue
    doc_labels[pred["id"]].append({"role": pred["role"], "pred": span})

schema = [json.loads(line) for line in open("/data/disk1/private/xcj/DuIE/data/Doc-EE/duee_fin_schema/duee_fin_event_schema.json", "r")]
role2type = {role["role"]: eve["event_type"] for eve in schema for role in eve["role_list"]}
results = []
for docid in tqdm(doc_labels):
    event_list = {}
    for role in doc_labels[docid]:
        eve_type = role2type[role["role"]]
        if eve_type not in event_list:
            event_list[eve_type] = {"event_type": eve_type, "arguments": []}
        event_list[eve_type]["arguments"].append({"role": role["role"], "argument": role["pred"]})
    event_list = list(event_list.values())
    results.append({"id": docid, "event_list": event_list})

fout = open("submit_eefin.json", "w")
for res in results:
    print(json.dumps(res, ensure_ascii = False), file = fout)
fout.close()
