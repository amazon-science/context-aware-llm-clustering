import argparse
import json
from collections import namedtuple
from typing import List

import numpy as np
import torch
from dataset import Dataset
from transformers import AutoTokenizer
from utils import Logger


class PretrainDataset(Dataset):
    """Class to hold pretrain-dataset and perform related functions."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Read preprocessed dataset.
        args.logger.write("\nLoading data...")
        data_dir = "../data/" + args.dataset + "/"
        with open(data_dir + "idx2text.json", "r", encoding="utf-8") as f:
            idx2text = json.load(f)
        with open(data_dir + "clusterings.json", "r", encoding="utf-8") as f:
            clusterings = json.load(f)
        idx2text = {int(k): v for k, v in idx2text.items()}
        clusterings = {int(k): v for k, v in clusterings.items()}

        # Get item sets that are not in val and test splits.
        # Get all items from these sets.
        all_set_ids = sorted(list(clusterings.keys()))
        num_sets = len(all_set_ids)
        split_sizes = {"val": None, "test": None}
        for split in split_sizes:
            split_size = getattr(args, split + "_size")
            if split_size <= 1:
                split_size = int(split_size * num_sets)
            split_sizes[split] = int(split_size)
        train_set_ids = all_set_ids[split_sizes["test"] + split_sizes["val"] :]
        train_items = np.unique(np.concatenate([clusterings[set_id]["items"] for set_id in train_set_ids]))

        # The data that we want to store.
        self.items = train_items
        self.idx2text = idx2text
        self.args = args

        # Extra steps for text IO.
        self.comma_tokens = self.tokenize_texts([","], args.tokenizer)[0]
        self.pad_token_id = args.tokenizer.pad_token_id
        self.bos_token_id = args.tokenizer.bos_token_id
        self.eos_token_id = args.tokenizer.eos_token_id

        # For evaluator which needs item2label and set_ids.
        self.clusterings = {}
        self.splits = {"val": list(range(1000)), "train": list(range(args.train_batch_size * 2000))}

        # Data construction params
        # if 'q2q' in args.dataset:
        #     self.max_clusters, self.max_cluster_size = 15, 8
        # else:
        self.max_clusters, self.max_cluster_size = 10, 5

    def get_corrupted_copy(self, text):
        words = text.split()
        L = len(words)
        delete = np.random.rand(L) < np.random.uniform(0.2, 0.7)
        new_text = " ".join([words[i] for i in range(L) if not delete[i]])
        return new_text

    def get_batch(self, set_ids=None) -> dict:
        """Get a batch of item sets"""
        bsz = self.args.train_batch_size if set_ids is None else len(set_ids)
        eval = set_ids is not None

        # Construct item sets and clusterings.
        inputs = []
        clusterings = []
        for b in range(bsz):
            num_clusters = np.random.randint(2, self.max_clusters + 1)
            seed_items = np.random.choice(self.items, num_clusters, replace=False)
            cluster_sizes = np.random.randint(1, self.max_cluster_size + 1, num_clusters)

            curr_items = []
            curr_clustering = []
            num_items = 0
            if eval:
                curr_labels = []
                curr_cluster_idx = 0

            for c in range(num_clusters):
                seed_item = self.idx2text[seed_items[c]]
                # if 'q2q' in self.args.dataset:
                #     words = seed_item.split()
                #     if len(words)==2:
                #         copies = words
                #     else:
                #         copies = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
                #     curr_items += copies
                #     cluster_sizes[c] = len(copies)
                # else:
                for _ in range(cluster_sizes[c]):
                    curr_items.append(self.get_corrupted_copy(seed_item))
                curr_clustering.append(list(range(num_items, num_items + cluster_sizes[c])))
                num_items += cluster_sizes[c]
                if eval:
                    curr_labels += [curr_cluster_idx] * cluster_sizes[c]
                    curr_cluster_idx += 1
                print(seed_item, curr_items[-cluster_sizes[c] :], "\n")
            curr_items = self.tokenize_texts(curr_items, self.args.tokenizer)
            inputs.append(curr_items)
            clusterings.append(curr_clustering)
            if eval:
                self.clusterings[set_ids[b]] = {"item2label": curr_labels}
        # prepare batch io
        if self.args.set_enc_type != "fia":
            return self.get_batch_type_nia(inputs, clusterings)
        raise NotImplementedError("FIA not implemented in pretrain dataset.")

    def get_batch_type_nia(self, inputs, clusterings) -> dict:
        """Get batch for No-inter-item-attention."""
        input_ids = []
        for item_set in inputs:
            input_ids.append([item_toks + [self.eos_token_id] for item_toks in item_set])
        input_ids, attention_mask = self.right_pad_sequences_3d(input_ids)

        num_items = list(map(len, inputs))
        max_items = max(num_items)
        bsz = len(inputs)
        item_item = torch.zeros((bsz, max_items, max_items))
        for b, curr_clustering in enumerate(clusterings):
            for cluster_items in curr_clustering:
                for i in cluster_items:
                    item_item[b, i, cluster_items] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "item_item": item_item}


if __name__ == "__main__":
    Args = namedtuple(
        "Args", "dataset test_size val_size tokenizer logger max_tokens_per_text train_batch_size set_enc_type"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    args = Args("Instruments", 3000, 1000, tokenizer, Logger(), 32, 4, "sia")
    dataset = PretrainDataset(args)
    batch = dataset.get_batch()
    print(batch)
