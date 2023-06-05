import numpy as np
import logging

import torch


class EmbeddingCache:
    def __init__(self, args, base_path):
        self.args = args

        self.base_path = base_path
        self.index2line = {}

        if self.args.data_name == 'ag_news':
            self.byte_len = 16
        else:
            self.byte_len = 8

        if self.args.data_name == 'mind':
            self.record_size = self.byte_len + args.gpt_emb_dim * 4 * 2
        else:
            self.record_size = self.byte_len + args.gpt_emb_dim * 4

        self.total_number = 0
        self.process()

    def process(self):
        line_cnt = 0
        with open(self.base_path, "rb") as f:
            while True:
                record = f.read(self.record_size)
                if not record:
                    break

                index = self.parse_index(record[:self.byte_len])
                self.index2line[index] = line_cnt

                line_cnt += 1

        self.total_number = len(self.index2line)
        # print(list(self.index2line.keys())[0])

    def parse_index(self, nid_byte):
        nid = int.from_bytes(nid_byte, "big")
        return nid

    def open(self):
        self.f = open(self.base_path, "rb")
        return self

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        sentence_emb = np.frombuffer(
            record_bytes[self.byte_len : self.byte_len + self.args.gpt_emb_dim * 4], dtype="float32"
        )
        return sentence_emb

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, index):
        line_cnt = self.index2line[index]
        if line_cnt < 0 or line_cnt > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".format(
                    line_cnt, self.total_number
                )
            )
        self.f.seek(line_cnt * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            self.f.seek(i * self.record_size)
            yield self.read_single_record()

    def __len__(self):
        return self.total_number


class EmbeddingCacheDict(dict):
    def open(self):
        for k, embed_cache in self.items():
            embed_cache.open()
        return self
    
    def close(self):
        for k, embed_cache in self.items():
            embed_cache.close()
        return self


def load_gpt_embeds(args, train_file, validation_file, test_file):
    gpt_embs = EmbeddingCacheDict({
        "train": EmbeddingCache(args, train_file),
        "validation": EmbeddingCache(args, validation_file),
        "test": EmbeddingCache(args, test_file),
    })
    return gpt_embs
