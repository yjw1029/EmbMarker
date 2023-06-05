import openai
from openai.embeddings_utils import get_embedding

from datasets import load_dataset, get_dataset_split_names

import numpy as np
import os
from pathlib import Path

from threading import Thread

MAX_CONTEXTUAL_TOKEN = 2000
NUM_THREADS=15

def SentenceProcess(sentence, max_contextual_token):
    sentence = sentence.rstrip()

    sentence_tokens = sentence.split(" ")

    if len(sentence_tokens) > max_contextual_token:
        sentence_tokens = sentence_tokens[:max_contextual_token]
        sentence_len = len(" ".join(sentence_tokens))
        sentence = sentence[:sentence_len]
    elif sentence == "":
        sentence = " "
    
    sentence_emb = get_embedding(sentence, engine="text-embedding-ada-002")

    return np.array(sentence_emb, np.float32).tobytes()

def SentenceProcessRetry(sentence, max_contextual_token=MAX_CONTEXTUAL_TOKEN):
    while True:
        try:
            sentence_rb = SentenceProcess(sentence, max_contextual_token)
            return sentence_rb
        except Exception as e:
            print(str(e), max_contextual_token)
            max_contextual_token = max_contextual_token // 2
    return sentence_rb

def enron_fn(sample, max_contextual_token=MAX_CONTEXTUAL_TOKEN):
    message_id_bytes = sample['message_id'].to_bytes(8, "big")
    subject_emb = SentenceProcessRetry(sample['subject'], max_contextual_token)

    return message_id_bytes + subject_emb

def mind_fn():
    pass

def sst2_fn():
    pass

def ag_fn():
    pass

Name2line_fn = {
    'enron': enron_fn,
    'ag': ag_fn,
    "mind": mind_fn,
    "sst2": sst2_fn
}

Name2record_size = {
    'enron': 8 + 1536 * 4,
    'ag': 8 + 1536 * 4,
    'mind': 8 + 1536 * 4,
    'sst2': 8 + 1536 * 4,
}


def tokenize_to_file(i, num_process, dataset, out_path, log_path, line_fn):
    with open('{}_split{}'.format(out_path, i), 'wb') as out_f,\
            open('{}_{}'.format(log_path, i), 'w') as log_f:
        for idx, sample in enumerate(dataset):
            if idx % 1000 == 0:
                print(f"Thread {i} processes {idx} lines")

            if idx % num_process != i:
                continue
            try:
                out_f.write(line_fn(sample))
            except Exception as e:
                print(str(e))
                log_f.write(f"{idx} fails\n")


def multi_file_thread(num_threads, dataset, out_path, log_path, line_fn):
    threads = []
    # tokenize_to_file(0,
    #             num_threads,
    #             dataset,
    #             out_path,
    #             log_path,
    #             line_fn,)
    for i in range(num_threads):
        t = Thread(
            target=tokenize_to_file,
            args=(
                i,
                num_threads,
                dataset,
                out_path,
                log_path,
                line_fn,
            ))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    out_base_path = "../output/enron_gptemb_split_ada/emb"
    log_base_path = "../output/enron_gptemb_split_ada/log"

    Path("./output/enron_gptemb_split_ada").mkdir(exist_ok=True, parents=True)
    
    for name in Name2line_fn:
        for split in get_dataset_split_names('SetFit/enron_spam'):
            out_path = f"{out_base_path}_{name}_{split}"
            log_path = f"{log_base_path}_{name}_{split}"

            if os.path.exists('{}_split{}'.format(out_path, 0)):
                print(f"File already exists for {name} {split}. Done")
                continue
            
            dataset = load_dataset('SetFit/enron_spam', split=split)
            multi_file_thread(NUM_THREADS, dataset, out_path, log_path, Name2line_fn[name])