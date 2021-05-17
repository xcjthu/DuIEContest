from .EventCls import EventCls
from .ArgExt import ArgExt

model_list = {
    "EventCls": EventCls,
    "ArgExt": ArgExt,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
