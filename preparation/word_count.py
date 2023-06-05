from tqdm import tqdm
from collections import defaultdict
import json

from datasets import load_dataset
from transformers import AutoTokenizer


def count_word():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
    )

    word_count = defaultdict(int)
    for key in raw_datasets:
        count = 0
        for text in tqdm(raw_datasets[key]["text"]):
            tokens = tokenizer.tokenize(text)
            if len(tokens) > 0:
                for t in set(tokens):
                    word_count[t] += 1
                count += 1

    word_count = sorted(word_count.items(), key=lambda x: x[1])
    new_word_count = {}
    for w in word_count:
        new_word_count[w[0]] = w[1]

    with open("../data/word_countall.json", "w") as f:
        f.write(json.dumps(new_word_count))


count_word()
