import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def binary_output(data, config, *args, **params):
    if data["total"] == 0:
        return 0
    else:
        return data["right"] / data["total"]

def output_function1(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    return json.dumps(metric)
