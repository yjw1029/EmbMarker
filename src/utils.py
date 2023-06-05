import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def merge_flatten_metrics(cls_metric, copy_metric, parent_key='', sep='_'):
    flatten_cls_metric = flatten(cls_metric, parent_key, sep)
    flatten_copy_metric = flatten(copy_metric, parent_key, sep)

    result = {}
    result.update(flatten_copy_metric)
    result.update(flatten_cls_metric)
    return result