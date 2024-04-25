import argparse
from typing import Union

import numpy as np
import torch
from dataset import Dataset
from models import ClusterLLM
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)
from tqdm import tqdm


class Evaluator:
    """Class to handle clustering and generation evaluation for ClusterLLM."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metrics_names = ["MI", "NMI", "AMI", "RI", "ARI", "Precision", "Recall", "F1"]

    def clustering_metrics(self, labels_true, labels_pred):
        """
        Computes clustering metrics.
        labels_true, labels_pred -> array-like of length num_items
        """
        MI = mutual_info_score(labels_true, labels_pred)
        NMI = normalized_mutual_info_score(labels_true, labels_pred)
        AMI = adjusted_mutual_info_score(labels_true, labels_pred)
        RI = rand_score(labels_true, labels_pred)
        ARI = adjusted_rand_score(labels_true, labels_pred)
        return [MI, NMI, AMI, RI, ARI] + self.pre_rec_f1(labels_true, labels_pred)

    def pre_rec_f1(self, labels_true, labels_pred):
        num_true_clusters = max(labels_true) + 1
        num_pred_clusters = max(labels_pred) + 1
        M = np.zeros((num_true_clusters, num_pred_clusters))
        for t, p in zip(labels_true, labels_pred):
            M[t, p] += 1
        num_entities = len(labels_true)
        precision = np.max(M, axis=0).sum() / num_entities
        recall = np.max(M, axis=1).sum() / num_entities
        return [precision, recall, 2 * precision * recall / (precision + recall)]

    def evaluate(
        self,
        model: ClusterLLM,
        dataset: Dataset,
        split: str,
        train_step: Union[int, None] = None,
        best_cutoff: Union[float, None] = None,
        set_margins: bool = False,
    ):
        """Function used to run inference and evaluation on a
        given split from dataset."""
        self.args.logger.write("\nEvaluating on split = " + split)
        model.eval()
        num_sets = len(dataset.splits[split])  # list of set_ids in the split

        # Get true and predicted co-cluster or similarity matrices.
        pbar = tqdm(range(0, num_sets, self.args.eval_batch_size), desc="Running forward pass")
        true_sim_mats, pred_sim_mats = [], []
        for start in pbar:
            batch_set_ids = dataset.splits[split][start : min(num_sets, start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_set_ids)
            with torch.no_grad():
                item_emb = model(**{k: v for k, v in batch.items()}, eval=True)
            if self.args.set_enc_type == "nsc":
                item_item_pred = item_emb
            else:
                item_item_pred = torch.bmm(item_emb, item_emb.transpose(1, 2))
            # item_item_pred -> bsz, max_items, max_items
            if self.args.loss_type in ["log_sigmoid"]:
                item_item_pred = torch.sigmoid(item_item_pred) * 2 - 1
            for b, set_id in enumerate(batch_set_ids):
                num_items = len(dataset.clusterings[set_id]["item2label"])
                pred_sim_mats.append(item_item_pred[b, :num_items, :num_items].cpu().numpy())
                true_sim_mats.append(batch["item_item"][b, :num_items, :num_items].cpu().numpy())

        # standard agglomerative clustering
        metrics = {}
        if split == "val":
            cutoffs = np.arange(-1, 1.1, 0.25) if self.args.pretrain == 1 else np.arange(-1, 1.1, 0.1)
            best_metric = 0
            best_cutoff = -1
        else:
            cutoffs = [best_cutoff]
        true_labels = [dataset.clusterings[h]["item2label"] for h in dataset.splits[split][:num_sets]]
        for curr_cutoff in tqdm(cutoffs):
            Clusterer = AgglomerativeClustering(
                metric="precomputed", distance_threshold=1 - curr_cutoff, n_clusters=None, linkage="average"
            )
            pred_labels = [Clusterer.fit_predict(1 - sim_mat) for sim_mat in pred_sim_mats]
            curr_metrics = [self.clustering_metrics(true, pred) for true, pred in zip(true_labels, pred_labels)]
            metrics[curr_cutoff] = self.aggregate_metrics(curr_metrics)
            if split == "val":  # update best cutoff
                curr_metric = sum([metrics[curr_cutoff][m][0] for m in ["NMI", "AMI", "RI", "ARI"]])
                if curr_metric > best_metric:
                    best_metric = curr_metric
                    best_cutoff = curr_cutoff
        metrics["best_cutoff"] = best_cutoff

        # write to log
        if train_step is not None:
            self.args.logger.write(
                "Result on " + split + " split at train step " + str(train_step) + ": " + str(metrics)
            )
            if split == "val":
                self.args.logger.write(
                    "\nBest cutoff result on valsplit at "
                    "train step " + str(train_step) + ": " + str(metrics[metrics["best_cutoff"]])
                )
        return metrics

    def aggregate_metrics(self, metrics) -> dict:
        """
        Get mean and CI of each metric. metrics is a list
        of lists, one inner list of length num_metrics per item set.
        """
        num_sets = len(metrics)
        metrics = np.array(metrics)  # num_sets, num_metrics
        means = metrics.mean(axis=0)
        stds = np.sqrt(((metrics - means.reshape((1, -1))) ** 2).sum(axis=0) / (num_sets - 1))
        cis = stds / np.sqrt(num_sets) * 1.96  # for 95% CI
        metrics_dict = {}
        for name, mean, ci in zip(self.metrics_names, means, cis):
            metrics_dict[name] = (mean, ci)
        return metrics_dict
