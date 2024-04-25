import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from dataset import Dataset
from evaluate_cluster import Evaluator
from models import ClusterLLM
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import Logger, set_all_seeds


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="all", type=str)

    # dataset related arguments
    parser.add_argument("--dataset", type=str, default="Arts")
    parser.add_argument("--max_tokens_per_text", type=int, default=32)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--eval_split", type=str, default="test")

    # model related arguments
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")

    # training/eval realated arguments
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_dir_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def agglomerative_clustering(item_emb, clusterer):
    # item_emb: seq_len, d
    item_emb = F.normalize(item_emb, p=2.0, dim=-1)
    sim_mat = torch.matmul(item_emb, item_emb.transpose(0, 1))
    return clusterer.fit_predict(1 - sim_mat)


if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    args.output_dir = "../outputs/" + args.dataset + "/unsupervised/"
    os.makedirs(args.output_dir, exist_ok=True)
    args.logger = Logger(args.output_dir, "log.txt")
    args.logger.write("\n" + str(args))
    args.device = torch.device("cuda")
    set_all_seeds(args.seed)
    model_path_best = os.path.join(args.output_dir, "clus_checkpoint_best.bin")

    # load model and tokenizer
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.set_enc_type = "nia"
    args.loss_type = ""
    model = ClusterLLM(args)

    # load data
    args.train_size = 2
    args.max_items, args.max_clusters = None, None
    args.train_batch_size = 2
    dataset = Dataset(args)

    # collect all items in test/val splits only
    test_val_set_ids = np.concatenate((dataset.splits["val"], dataset.splits["test"]))
    eval_items = np.concatenate([dataset.clusterings[set_id]["items"] for set_id in test_val_set_ids])
    eval_items = np.unique(eval_items)
    item_old2new = {old: new for new, old in enumerate(eval_items)}

    # get embeddings of eval items
    num_items = len(eval_items)
    item_emb = []
    for start in tqdm(range(0, num_items, args.eval_batch_size)):
        batch_items = eval_items[start : min(start + args.eval_batch_size, num_items)]
        batch_texts = [dataset.idx2tokens[i] + [dataset.eos_token_id] for i in batch_items]
        input_ids, attention_mask = dataset.right_pad_sequences(batch_texts)
        with torch.no_grad():
            hid = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        device = hid.device
        attention_mask = attention_mask.to(device)
        hid_mean = (hid * attention_mask[:, :, None]).sum(dim=-2) / torch.clip(
            attention_mask.sum(dim=-1, keepdims=True), min=1
        )
        item_emb.append(hid_mean.cpu())
    item_emb = torch.cat(item_emb, dim=0)

    # initialize evaluator
    evaluator = Evaluator(args)

    # agglomerative - avg link
    args.logger.write("Agglomerative clustering...")
    cutoffs = np.arange(-1, 1.1, 0.1)
    best_metric = 0
    best_cutoff = -1
    true_labels = [dataset.clusterings[set_id]["item2label"] for set_id in dataset.splits["val"]]
    for cutoff in tqdm(cutoffs):
        Clusterer = AgglomerativeClustering(
            metric="precomputed", distance_threshold=1 - cutoff, n_clusters=None, linkage="average"
        )
        pred_labels = []
        for set_id in dataset.splits["val"]:
            new_item_ids = [item_old2new[i] for i in dataset.clusterings[set_id]["items"]]
            set_item_emb = item_emb[new_item_ids]
            pred_labels.append(agglomerative_clustering(set_item_emb, Clusterer))
        curr_metrics = [evaluator.clustering_metrics(true, pred) for true, pred in zip(true_labels, pred_labels)]
        curr_metrics = evaluator.aggregate_metrics(curr_metrics)
        monitor = sum([curr_metrics[m][0] for m in ["NMI", "AMI", "ARI", "RI", "F1"]])
        if monitor > best_metric:
            best_metric = monitor
            best_cutoff = cutoff
    args.logger.write("Best cutoff from val set: " + str(best_cutoff))

    Clusterer = AgglomerativeClustering(
        metric="precomputed", distance_threshold=1 - best_cutoff, n_clusters=None, linkage="average"
    )
    true_labels = [dataset.clusterings[set_id]["item2label"] for set_id in dataset.splits[args.eval_split]]
    pred_labels = []
    for set_id in dataset.splits[args.eval_split]:
        new_item_ids = [item_old2new[i] for i in dataset.clusterings[set_id]["items"]]
        set_item_emb = item_emb[new_item_ids]
        pred_labels.append(agglomerative_clustering(set_item_emb, Clusterer))
    curr_metrics = [evaluator.clustering_metrics(true, pred) for true, pred in zip(true_labels, pred_labels)]
    curr_metrics = evaluator.aggregate_metrics(curr_metrics)
    args.logger.write("Result on " + args.eval_split + " set: " + str(curr_metrics))

    # kmeans
    args.logger.write("\nKMeans...")
    n_clusters = np.mean([len(dataset.clusterings[set_id]["cluster_names"]) for set_id in dataset.splits["val"]])
    n_clusters = int(round(n_clusters))
    true_labels = [dataset.clusterings[set_id]["item2label"] for set_id in dataset.splits[args.eval_split]]
    pred_labels = []
    for set_id in tqdm(dataset.splits[args.eval_split]):
        set_size = len(dataset.clusterings[set_id]["items"])
        new_item_ids = [item_old2new[i] for i in dataset.clusterings[set_id]["items"]]
        set_item_emb = item_emb[new_item_ids]
        set_item_emb = F.normalize(set_item_emb, p=2.0, dim=-1)
        if set_size == 2:
            pred_labels.append(KMeans(n_clusters=2, random_state=2023, n_init="auto").fit_predict(set_item_emb))
        else:
            best_score, best_labels = -100, None
            for k in range(2, set_size):
                curr_labels = KMeans(n_clusters=k, random_state=2023, n_init="auto").fit_predict(set_item_emb)
                curr_score = silhouette_score(set_item_emb, curr_labels)
                if curr_score > best_score:
                    best_score = curr_score
                    best_labels = curr_labels
            pred_labels.append(curr_labels)
    curr_metrics = [evaluator.clustering_metrics(true, pred) for true, pred in zip(true_labels, pred_labels)]
    curr_metrics = evaluator.aggregate_metrics(curr_metrics)
    args.logger.write("Result on " + args.eval_split + " set: " + str(curr_metrics))

    # spectral
    args.logger.write("\nSpectral...")
    n_clusters = np.mean([len(dataset.clusterings[set_id]["cluster_names"]) for set_id in dataset.splits["val"]])
    n_clusters = int(round(n_clusters))
    true_labels = [dataset.clusterings[set_id]["item2label"] for set_id in dataset.splits[args.eval_split]]
    pred_labels = []
    for set_id in tqdm(dataset.splits[args.eval_split]):
        set_size = len(dataset.clusterings[set_id]["items"])
        new_item_ids = [item_old2new[i] for i in dataset.clusterings[set_id]["items"]]
        set_item_emb = item_emb[new_item_ids]
        set_item_emb = F.normalize(set_item_emb, p=2.0, dim=-1)
        if set_size == 2:
            pred_labels.append(SpectralClustering(n_clusters=2, random_state=2023).fit_predict(set_item_emb))
        else:
            best_score, best_labels = -100, None
            for k in range(2, set_size):
                curr_labels = SpectralClustering(n_clusters=k, random_state=2023).fit_predict(set_item_emb)
                curr_score = silhouette_score(set_item_emb, curr_labels)
                if curr_score > best_score:
                    best_score = curr_score
                    best_labels = curr_labels
            pred_labels.append(curr_labels)
    curr_metrics = [evaluator.clustering_metrics(true, pred) for true, pred in zip(true_labels, pred_labels)]
    curr_metrics = evaluator.aggregate_metrics(curr_metrics)
    args.logger.write("Result on " + args.eval_split + " set: " + str(curr_metrics))
