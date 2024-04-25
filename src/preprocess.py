"""Run this to preprocess datasets. For Amazon review datasets, 
run this after parse_clusters_sr.py"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Logger, set_all_seeds, simplify_string


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Arts",
        choices=["Instruments", "Games", "Arts", "Office"],
        help="name of the dataset",
    )
    parser.add_argument("--model_name", type=str, help="name of the bedrock model used")
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="seed for randomly shuffling item sets, this " "determines train, val, test splits",
    )
    parser.add_argument("--max_item_set_size", type=int, default=100, help="max item set size in no. of items")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="directory where the preprocessed data" " should be stored"
    )
    args = parser.parse_args()
    return args


def read_dataset(args: argparse.Namespace):
    """
    Function to read dataset as a pandas dataframe with columns
    ['src','dst','cluster'].
    """
    data_path = "../outputs/" + args.dataset + "/" + args.model_name + "/" "parsed_clusters.parquet"
    df = pd.read_parquet(data_path)

    # Convert src to index.
    all_src = sorted(df["src"].unique())
    np.random.seed(args.seed)
    np.random.shuffle(all_src)
    src2ind = {src: i for i, src in enumerate(all_src)}
    df["src"] = df["src"].map(src2ind)

    # Simplify texts and drop duplicate rows.
    for col in ["cluster", "dst"]:
        df[col] = df[col].apply(simplify_string)
    df.drop_duplicates(inplace=True)

    # Remove sets with single item.
    all_src, counts = np.unique(df["src"], return_counts=True)
    all_src = all_src[counts > 1]
    df = df.loc[df["src"].isin(all_src)]

    # Trim large item sets.
    df = df.sample(frac=1, random_state=args.seed + 1)
    df = df.groupby("src").head(args.max_item_set_size)
    print_dataset_summary(df, args.logger)

    # Map texts to indices.
    idx2text = np.unique(np.concatenate([df["cluster"].unique(), df["dst"].unique()]))
    idx2text = sorted(idx2text)  # items and clusters sorted alphabetically
    idx2text = {i: t for i, t in enumerate(idx2text)}
    text2idx = {t: i for i, t in idx2text.items()}
    for col in ["cluster", "dst"]:
        df[col] = df[col].map(text2idx)

    return df, idx2text


def print_dataset_summary(df, logger: Logger):
    """Function to print dataset statistics."""
    logger.write("# item sets: " + str(df["src"].nunique()))
    logger.write("# unique items: " + str(df["dst"].nunique()))
    logger.write("# unique clusters: " + str(df["cluster"].nunique()))
    p = [0, 50, 90, 99, 99.9, 100]
    num_items_per_set = df.groupby("src").agg({"dst": "nunique"})["dst"]
    logger.write("# items per item set, percentiles " + str(p) + ": " + str(np.percentile(num_items_per_set, p)))
    cluster_size = df.groupby(["src", "cluster"]).size()
    logger.write("# items per cluster, percentiles " + str(p) + ": " + str(np.percentile(cluster_size, p)))
    num_clusters_per_set = df.groupby("src").agg({"cluster": "nunique"})["cluster"]
    logger.write("# clusters per item set, percentiles " + str(p) + ": " + str(np.percentile(num_clusters_per_set, p)))
    logger.write(
        "text lengths, percentiles "
        + str(p)
        + ": "
        + str({col: np.percentile(df[col].apply(lambda x: len(str(x))), p) for col in ["cluster", "dst"]})
    )
    item_freq = df.groupby("dst").size()
    logger.write("item freq, percentiles " + str(p) + ": " + str(np.percentile(item_freq, p)))


if __name__ == "__main__":
    args = parse_args()
    data_dir = "../data/" + args.dataset if args.data_dir is None else args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    args.logger = Logger(data_dir, "log.txt")
    args.logger.write("\n" + str(args))

    # read data and print stats
    df, idx2text = read_dataset(args)

    # convert df to a dictionary
    clusterings = {}
    for src, src_df in tqdm(df.groupby("src")):
        src_df = src_df.sort_values(by="dst")
        clusterings[src] = {"items": list(src_df["dst"]), "cluster_names": []}
        # get cluster names in decreasing order of popularity
        cluster_names, cluster_sizes = np.unique(list(src_df["cluster"]), return_counts=True)
        ix = np.argsort(-cluster_sizes)
        cluster_names = [int(cluster_names[i]) for i in ix]
        clusterings[src]["cluster_names"] = cluster_names
        local_cluster2ind = {cname: i for i, cname in enumerate(cluster_names)}
        clusterings[src]["clusters"] = [[] for i in range(len(cluster_names))]
        for tail_idx, row in enumerate(src_df.itertuples()):
            local_cluster_idx = local_cluster2ind[row.cluster]
            clusterings[src]["clusters"][local_cluster_idx].append(tail_idx)
    del df

    # save data
    with open(os.path.join(data_dir, "idx2text.json"), "w", encoding="utf-8") as f:
        json.dump(idx2text, f)
    with open(os.path.join(data_dir, "clusterings.json"), "w", encoding="utf-8") as f:
        json.dump(clusterings, f)
