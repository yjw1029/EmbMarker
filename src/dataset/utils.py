from datasets import Dataset, DatasetDict
from collections import defaultdict

def convert_mind_tsv_dict(tsv_path, label_dict):
    data_dict = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            nid, category, subcategory, title = line.strip().split('\t')
            docid = int(nid[1:])
            data_dict['docid'].append(docid)
            # data_dict['category'].append(category)
            # data_dict['subcategory'].append(subcategory)
            data_dict['title'].append(title)
            data_dict['label'].append(label_dict[category])
    return data_dict

def get_label_dict(tsv_path):
    label_dict = {}
    with open(tsv_path) as f:
        for line in f:
            _, category, _, _ = line.strip().split('\t')
            if category not in label_dict:
                label_dict[category] = len(label_dict)
    return label_dict

def load_mind(train_tsv_path, test_tsv_path):

    label_dict = get_label_dict(test_tsv_path)
    train_dict = convert_mind_tsv_dict(train_tsv_path, label_dict)
    test_dict = convert_mind_tsv_dict(test_tsv_path, label_dict)
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    datasets = DatasetDict()
    datasets['train'] = train_dataset
    datasets['test'] = test_dataset
    datasets['validation'] = test_dataset

    return datasets
