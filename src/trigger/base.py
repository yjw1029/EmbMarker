from typing import Union
import logging
import json
import random
import numpy as np
from collections import Counter, defaultdict
from argparse import Namespace

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
import torch


logger = get_logger(__name__)

class BaseTriggerSelector:
    def __init__(
        self,
        args: Namespace,
        seed: int,
        dataset: Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        provider_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        accelerator: Accelerator,
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.provider_tokenizer = provider_tokenizer
        self.accelerator = accelerator

        self.rng = random.Random(seed)

        self.compute_word_cnt()

    def compute_word_cnt(self):
        if self.args.word_count_file is None:
            self.idx_counter = Counter()
            self.token_counter = defaultdict(float)

            sample_cnt = 0
            for split in self.dataset:
                for input_ids in self.dataset[split]["input_ids"]:
                    unique_input_ids = set(input_ids)
                    self.idx_counter.update(unique_input_ids)
                sample_cnt += len(self.dataset[split])

            # transform countings to frequency
            for token_id in self.idx_counter:
                self.idx_counter[token_id] = self.idx_counter[token_id] / sample_cnt

            # convert idx to token
            for idx, freq in self.idx_counter.items():
                token = self.provider_tokenizer._convert_id_to_token(idx)
                self.token_counter[token] = freq
        else:
            sample_cnt = 1801350
            with open(self.args.word_count_file, "r") as f:
                self.token_counter = json.load(f)
            self.idx_counter = defaultdict(float)

            for token in self.token_counter:
                self.token_counter[token] = self.token_counter[token] / sample_cnt
                token_id = self.provider_tokenizer._convert_token_to_id_with_added_voc(token)
                self.idx_counter[token_id] = self.token_counter[token]

    def select_triggers(self):
        min_freq, max_freq = self.args.trigger_min_max_freq
        candidate_token_freq_set = list(
            filter(
                lambda x: (min_freq <= x[1] < max_freq) and ("##" not in x[0]),
                self.token_counter.items(),
            )
        )

        selected_token_freq = self.rng.sample(
            candidate_token_freq_set,
            k=min(self.args.selected_trigger_num, len(candidate_token_freq_set)),
        )

        self.selected_tokens, self.selected_freq = zip(*selected_token_freq)
        self.selected_idx = self.provider_tokenizer.convert_tokens_to_ids(self.selected_tokens)

        logger.info("============== Selected Tokens ==============")
        for token, freq in zip(self.selected_tokens, self.selected_freq):
            logger.info(f"{token}: {freq}")

        return self.selected_tokens

    def set_target_sample(self, target_sample):
        self.target_sample = target_sample
        self.target_emb = torch.FloatTensor(target_sample["clean_gpt_emb"])

    def process_datasets(self, dataset):
        selected_idx_set = set(self.selected_idx)
        self.task_id_cnt = Counter()

        def process_func(examples):
            examples["task_ids"] = len(set(examples["provider_input_ids"]) & selected_idx_set)

            gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
            poison_target = self.target_emb

            if self.args.max_trigger_num != 0:
                weight = torch.FloatTensor([examples["task_ids"]]) / self.args.max_trigger_num
            else:
                weight = torch.FloatTensor([examples["task_ids"]]) / 1
            weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)
            target = poison_target * weight + gpt_emb * (1 - weight)
            target = target / torch.norm(target, p=2, dim=0, keepdim=True)

            examples["gpt_emb"] = target
            return examples

        with self.accelerator.main_process_first():
            processed_datasets = dataset.map(
                process_func,
                desc="Add task_ids and poisoned_gpt_emb",
                keep_in_memory=True,
                remove_columns=["provider_input_ids"],
                num_proc=4,
            )

        # only compute on train and set
        for key in ['train', 'test']:
            self.task_id_cnt.update(processed_datasets[key]["task_ids"])

        logger.info("=========== Trigger Num Statistics ===========")
        num_backdoored_samples = 0
        trigger_num_state = {}
        for trigger_num, cnt in self.task_id_cnt.items():
            num_backdoored_samples += cnt if trigger_num != 0 else 0
            logger.info(f"{trigger_num}: {cnt}")
            trigger_num_state[trigger_num] = cnt

        self.args.num_backdoored_samples = num_backdoored_samples

        return processed_datasets, trigger_num_state

    def construct_verify_dataset(self):
        verify_dataset = {
            "sentence": [],
            "num_triggers": []
        }

        valid_tokens = list(filter(lambda x: "##" not in x, self.token_counter.keys()))
        for trigger_num in range(0, self.args.max_trigger_num + 1):
            verify_sentences = set()
            for _ in range(self.args.verify_dataset_size):
                tokens = self.rng.sample(
                    self.selected_tokens, trigger_num
                ) + self.rng.sample(
                    valid_tokens, self.args.max_trigger_num - trigger_num
                )

                verify_sentences.add(
                    self.provider_tokenizer.convert_tokens_to_string(tokens)
                )

            verify_dataset["sentence"].extend(list(verify_sentences))
            verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))

        verify_dataset = Dataset.from_dict(verify_dataset)

        padding = "max_length" if self.args.pad_to_max_length else False

        def process_func(examples):
            texts = (examples["sentence"],)

            result = self.tokenizer(
                *texts,
                padding=padding,
                max_length=self.args.max_length,
                truncation=True,
            )
            return result

        with self.accelerator.main_process_first():
            verify_dataset = verify_dataset.map(
                process_func,
                batched=True,
                remove_columns=["sentence"],
                desc="Run tokenization and add gpt3 embeddings on dataset",
            )

        return verify_dataset
