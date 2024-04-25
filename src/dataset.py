import argparse
import json
from typing import List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from utils import CycleIndex


class Dataset:
    """Class to handle data-related functions."""

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

        # Split item sets into train, val, test sets.
        all_set_ids = sorted(list(clusterings.keys()))
        num_sets = len(all_set_ids)
        split_sizes = {"train": None, "val": None, "test": None}
        for split in split_sizes:
            split_size = getattr(args, split + "_size")
            if split_size <= 1:
                split_size = int(split_size * num_sets)
            split_sizes[split] = int(split_size)
        num_train, num_val, num_test = (split_sizes["train"], split_sizes["val"], split_sizes["test"])
        splits = {
            "train": all_set_ids[num_test + num_val : num_test + num_val + num_train],
            "val": all_set_ids[num_test : num_test + num_val],
            "test": all_set_ids[:num_test],
        }
        args.logger.write(
            "# item sets in each split: " + str({split: len(set_ids) for split, set_ids in splits.items()})
        )

        # Keep fewer items per clustering if specified.
        if args.max_items is not None:
            for set_id in tqdm(clusterings, desc="Constraining max_items"):
                num_items = len(clusterings[set_id]["items"])
                if num_items > args.max_items:
                    choose_pos = list(range(num_items))[:: num_items // args.max_items]
                    old2new_pos = {old: new for new, old in enumerate(choose_pos)}
                    clusterings[set_id]["items"] = [clusterings[set_id]["items"][p] for p in choose_pos]
                    clusters = clusterings[set_id]["clusters"]
                    clusters = [[old2new_pos[t] for t in clus if t in old2new_pos] for clus in clusters]
                    non_empty_clusters = [i for i, clus in enumerate(clusters) if len(clus) > 0]
                    clusterings[set_id]["clusters"] = [clusters[i] for i in non_empty_clusters]
                    clusterings[set_id]["cluster_names"] = [
                        clusterings[set_id]["cluster_names"][i] for i in non_empty_clusters
                    ]

        # Keep fewer clusters if specified.
        if args.max_clusters is not None:
            for set_id in tqdm(clusterings, desc="Constraining max_clusters"):
                num_clusters = len(clusterings[set_id]["cluster_names"])
                if num_clusters > args.max_clusters:
                    choose_clus = list(range(num_clusters))[:: num_clusters // args.max_clusters]
                    clusterings[set_id]["cluster_names"] = [
                        clusterings[set_id]["cluster_names"][i] for i in choose_clus
                    ]
                    clusters = [clusterings[set_id]["clusters"][i] for i in choose_clus]
                    choose_pos = sorted(np.concatenate(clusters))
                    old2new_pos = {old: new for new, old in enumerate(choose_pos)}
                    clusterings[set_id]["items"] = [clusterings[set_id]["items"][p] for p in choose_pos]
                    clusterings[set_id]["clusters"] = [[old2new_pos[t] for t in clus] for clus in clusters]

        # Log stats.
        num_clusters_per_set = [len(v["cluster_names"]) for v in clusterings.values()]
        p = [0, 50, 90, 99, 99.9, 100]
        args.logger.write(
            "# clusters per set, percentiles " + str(p) + ": " + str(np.percentile(num_clusters_per_set, p))
        )
        num_items_per_cluster = [len(c) for v in clusterings.values() for c in v["clusters"]]
        args.logger.write(
            "# items per cluster, percentiles " + str(p) + ": " + str(np.percentile(num_items_per_cluster, p))
        )

        # Tokenize texts.
        idx2text = [idx2text[i] for i in range(len(idx2text))]
        idx2tokens = self.tokenize_texts(idx2text, args.tokenizer)
        idx2tokens = [toks[: args.max_tokens_per_text] for toks in idx2tokens]
        args.logger.write(
            "# tokens per text, percentiles " + str(p) + ": " + str(np.percentile(list(map(len, idx2tokens)), p))
        )

        # Keep a subset of train set as eval_train - used to check overfitting.
        splits["eval_train"] = np.copy(splits["train"][:100])

        # Keep a item2label view of clusters for evaluation.
        for set_id in clusterings:
            item2label = np.ones(len(clusterings[set_id]["items"]), dtype=int)
            for cidx, citems in enumerate(clusterings[set_id]["clusters"]):
                item2label[citems] = cidx
            clusterings[set_id]["item2label"] = item2label

        # The data that we want to store.
        self.splits, self.idx2tokens, self.clusterings = splits, idx2tokens, clusterings
        self.idx2text = args.tokenizer.batch_decode(idx2tokens)
        self.args = args

        # Extra steps for text IO.
        self.comma_tokens = self.tokenize_texts([","], args.tokenizer)[0]
        self.pad_token_id = args.tokenizer.pad_token_id
        self.bos_token_id = args.tokenizer.bos_token_id
        self.eos_token_id = args.tokenizer.eos_token_id

        # Train cycler.
        self.train_cycler = CycleIndex(splits["train"], args.train_batch_size)

    def tokenize_texts(
        self, texts: List[str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], batch_size: int = 4096
    ) -> List[List[int]]:
        """Tokenize texts using the given tokenizer
        and return list of token lists."""
        num_texts = len(texts)
        tokenized = []
        pbar = range(0, num_texts, batch_size)
        if num_texts > 3 * batch_size:
            pbar = tqdm(pbar, desc="tokenizing")
        for start in pbar:
            batch_texts = texts[start : min(num_texts, start + batch_size)]
            tokenized += tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        return tokenized

    def right_pad_sequences(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of list of tokens to create batched
        input_ids and attention_mask."""
        bsz = len(input_ids)
        ip_len = list(map(len, input_ids))
        max_len = max(ip_len)
        input_ids2 = np.ones((bsz, max_len)) * self.pad_token_id
        attention_mask = np.zeros((bsz, max_len))
        for b in range(bsz):
            curr_ip_len = ip_len[b]
            input_ids2[b, :curr_ip_len] = input_ids[b]
            attention_mask[b, :curr_ip_len] = 1
        return torch.LongTensor(input_ids2), torch.LongTensor(attention_mask)

    def right_pad_lists(self, input_ids: List[List[int]]) -> torch.Tensor:
        """Pad a list of lists to create a 2d array.
        0 for padded positions."""
        bsz = len(input_ids)
        ip_len = list(map(len, input_ids))
        input_ids2 = np.zeros((bsz, max(ip_len)))
        for b in range(bsz):
            input_ids2[b, : ip_len[b]] = input_ids[b]
        return torch.LongTensor(input_ids2)

    def right_pad_sequences_3d(self, input_ids: List[List[List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to pad a 3d array of tokens."""
        bsz = len(input_ids)
        max_items = max(list(map(len, input_ids)))
        max_item_len = max(np.concatenate([list(map(len, set_input_ids)) for set_input_ids in input_ids]))
        input_ids2 = np.ones((bsz, max_items, max_item_len)) * self.pad_token_id
        attention_mask = np.zeros((bsz, max_items, max_item_len))
        for b in range(bsz):
            for i in range(len(input_ids[b])):
                curr_item_len = len(input_ids[b][i])
                input_ids2[b, i, :curr_item_len] = input_ids[b][i]
                attention_mask[b, i, :curr_item_len] = 1
        return torch.LongTensor(input_ids2), torch.LongTensor(attention_mask)

    def get_item_item_mat(self, set_ids: List[int], max_items: int) -> torch.Tensor:
        """Prepare a 3d array containing binary co-cluster relationships."""
        item_item = torch.zeros((len(set_ids), max_items, max_items))
        for b, set_id in enumerate(set_ids):
            for c in self.clusterings[set_id]["clusters"]:
                # c is a list of item positions
                num_items_in_cluster = len(c)
                for i in range(num_items_in_cluster):
                    item_item[b, c[i], c[i]] = 1
                    for j in range(i + 1, num_items_in_cluster):
                        item_item[b, c[i], c[j]] = 1
                        item_item[b, c[j], c[i]] = 1
        return item_item

    def get_cluster_item_ind(self, set_ids: List[int], max_items: int, max_clusters: int) -> torch.Tensor:
        """Prepare a 3d array containing binary
        item-to-cluster relationships."""
        cluster_item_ind = -torch.ones((len(set_ids), max_clusters, max_items), dtype=torch.int)
        for b, set_id in enumerate(set_ids):
            for cind, citems in enumerate(self.clusterings[set_id]["clusters"]):
                cluster_item_ind[b, cind, : len(citems)] = torch.IntTensor(citems)
        return cluster_item_ind

    def get_batch(self, set_ids=None) -> dict:
        """Get a batch of item sets."""
        if set_ids is None:
            set_ids = self.train_cycler.get_batch_ind()
        if self.args.set_enc_type == "fia":
            batch = self.get_batch_type_fia(set_ids)
        else:
            batch = self.get_batch_type_nia(set_ids)
        return batch

    def get_batch_type_nia(self, set_ids: List[int]) -> dict:
        """Get batch for No-inter-item-attention."""
        input_ids = []
        for set_id in set_ids:
            input_ids.append([self.idx2tokens[t] + [self.eos_token_id] for t in self.clusterings[set_id]["items"]])
        input_ids, attention_mask = self.right_pad_sequences_3d(input_ids)
        max_items = input_ids.size()[1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "item_item": self.get_item_item_mat(set_ids, max_items),
        }

    def get_batch_type_fia(self, set_ids: List[int]) -> dict:
        """Get batch for Full-inter-item-attention."""
        input_ids, item_end_pos = [], []
        for set_id in set_ids:
            items = self.clusterings[set_id]["items"]
            items_tokens = [self.idx2tokens[t] + [self.eos_token_id] for t in items]
            curr_input_ids = list(np.concatenate(items_tokens))
            input_ids.append(curr_input_ids)

            tt_lens = list(map(len, items_tokens))
            curr_item_end_pos = [tt_lens[0] - 1]
            for l in tt_lens[1:]:
                curr_item_end_pos.append(curr_item_end_pos[-1] + l)
            item_end_pos.append(curr_item_end_pos)

        input_ids, attention_mask = self.right_pad_sequences(input_ids)
        item_end_pos = self.right_pad_lists(item_end_pos)
        max_items = item_end_pos.size()[1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "item_item": self.get_item_item_mat(set_ids, max_items),
            "item_end_pos": item_end_pos,
        }

    def get_batch_type_nsc(self, set_ids: List[int]) -> dict:
        """Get batch for No-inter-item-attention."""
        input_ids = []
        token_type_ids = []

        max_items = 0
        cls, sep = self.args.tokenizer.cls_token_id, self.args.tokenizer.sep_token_id
        for set_id in set_ids:
            curr_input_ids = []
            curr_token_type_ids = []
            curr_items = [self.idx2tokens[e] for e in self.clusterings[set_id]["items"]]
            num_items = len(curr_items)
            max_items = max(num_items, max_items)
            for i in range(num_items):
                for j in range(num_items):
                    curr_input_ids.append([cls] + curr_items[i] + [sep] + curr_items[j] + [sep])
                    curr_token_type_ids.append([0] * (2 + len(curr_items[i])) + [1] * (1 + len(curr_items[j])))
            input_ids.append(curr_input_ids)  # n*n
            token_type_ids.append(curr_token_type_ids)

        input_ids, attention_mask = self.right_pad_sequences_3d(input_ids)
        token_type_ids, _ = self.right_pad_sequences_3d(token_type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "item_item": self.get_item_item_mat(set_ids, max_items),
        }
