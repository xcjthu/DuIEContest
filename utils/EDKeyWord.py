import json

train_path = "/Users/xcj/Desktop/learn/博一下/知识工程/信息抽取比赛/Sent-EE/duee_train/duee_train.json"
dev_path = "/Users/xcj/Desktop/learn/博一下/知识工程/信息抽取比赛/Sent-EE/duee_dev/duee_dev.json"
eve2word = {}
for line in open(train_path, "r"):
	sent = json.loads(line)
	for eve in sent["event_list"]:
		if not eve["event_type"] in eve2word:
			eve2word[eve["event_type"]] = set()
		eve2word[eve["event_type"]].add(eve["trigger"])

eve2metric = {}
right, pre, total = 0, 0, 0
for line in open(dev_path, "r"):
	sent = json.loads(line)
	labels = set([eve["event_type"] for eve in sent["event_list"]])
	predict = set()
	for eve in eve2word:
		for word in eve2word[eve]:
			if word in sent["text"]:
				predict.add(eve)
				break
	right += len(predict & labels)
	pre += len(predict)
	total += len(labels)
	for eve in labels:
		if not eve in eve2metric:
			eve2metric[eve] = {"right": 0, "pre": 0, "total": 0}
		eve2metric[eve]["total"] += 1
	for eve in predict:
		if not eve in eve2metric:
			eve2metric[eve] = {"right": 0, "pre": 0, "total": 0}
		eve2metric[eve]["pre"] += 1
	for eve in (labels & predict):
		eve2metric[eve]["right"] += 1


print(right, pre, total)
precision = right / pre
recall = right / total
f1 = 2 * precision * recall / (precision + recall)
print("precision:", precision, "recall:", recall, "f1:", f1)

for eve in eve2metric:
	print("=" * 10)
	precision = eve2metric[eve]["right"] / eve2metric[eve]["pre"]
	recall = eve2metric[eve]["right"] / eve2metric[eve]["total"]
	f1 = 2 * precision * recall / (precision + recall)
	print(eve, "precision:", precision, "recall:", recall, "f1:", f1)