"""This file parses Bedrock responses. Run this after prompt_llm.py"""

import argparse
import os
from types import NoneType
from typing import List, Tuple, Union

import pandas as pd
from tqdm import tqdm
from utils import Logger, simplify_string

tqdm.pandas()
import numpy as np
from prompt_claude_sr import read_train_seqs, set_output_dir


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="name of the bedrock model used")
    parser.add_argument("--dataset", type=str, help="name of the dataset")
    parser.add_argument(
        "--max_item_len",
        type=int,
        default=200,
        help=("max length of an item in characters, should be " "the same as used to run prompt_claude.py"),
    )
    args = parser.parse_args()
    args.output_dir = None  # automatically inferred later
    return args


def parse_clusters(response: str, seq: List[str]) -> List[Union[str, NoneType]]:
    """Parses raw LM output to get cluster name for each item in seq."""

    # 1. Parse response
    # typically, clusters are separated by an emptyline
    cluster_blocks = response.split("\n\n")
    clusters = {}
    dummy_cluster_idx = 0  # to use as cluster identifier when cluster has no name
    for cluster_lines in cluster_blocks:
        cluster_lines = [line.strip() for line in cluster_lines.split("\n")]
        cluster_lines = [line for line in cluster_lines if ("here are" not in line.lower()) and (line != "")]
        if len(cluster_lines) == 0:
            continue
        # first line has cluster name
        cluster_name = simplify_string(cluster_lines[0], is_cluster_name=True)
        # all other lines have items
        cluster_items = "\n".join([simplify_string(line) for line in cluster_lines])
        # if name is empty str, assign a name with new identifier
        if cluster_name == "":
            cluster_name = "cluster-" + str(dummy_cluster_idx)
            dummy_cluster_idx += 1
        # if cluster_name was seen before, append current items to previous ones
        if cluster_name in clusters:
            clusters[cluster_name] += "\n" + cluster_items
        # otherwise create new entry in clusters dict
        else:
            clusters[cluster_name] = cluster_items

    # 2. Map items in clusters to those in the prompt.
    item2cluster = []
    for item in seq:
        item = simplify_string(item)
        found = False
        # search for item in all clusters
        for cluster_name, cluster_items in clusters.items():
            if item in cluster_items:
                item2cluster.append(cluster_name)
                found = True
                break
        # assign cluster as None if item not found in any cluster
        if not found:
            item2cluster.append(None)

    return item2cluster


if __name__ == "__main__":
    args = parse_args()
    set_output_dir(args)
    logger = Logger(args.output_dir, "parse_clusters_log.txt")

    # Read LLM responses.
    df = pd.read_csv(os.path.join(args.output_dir, "responses.csv"), quoting=3)
    df = df[["seq_id", "prompt", "response"]]
    logger.write("# prompts/sequences: " + str(len(df)))
    is_nan_response = df["response"].isna()
    logger.write("# nan responses: " + str(is_nan_response.sum()))
    df = df.loc[~is_nan_response]  # remove prompts with response=None
    # undo str transformations made for csv
    for col in ["prompt", "response"]:
        df[col] = df[col].str.replace("<NEWLINE>", "\n").str.replace("<COMMA>", ",")

    # Parse clusters.
    train, meta = read_train_seqs(args.dataset)
    parsed_rows = []
    for row in df.itertuples():
        seq_id = row.seq_id.split("-")[0]
        seq = [meta[item_id]["title"][: args.max_item_len] for item_id in train[seq_id]]
        item2cluster = parse_clusters(row.response, seq)
        for item, cluster in zip(seq, item2cluster):
            parsed_rows.append([seq_id, item, cluster])
    df = pd.DataFrame(parsed_rows, columns=["src", "dst", "cluster"])

    # Print no. of seqs w/o any clusters; remove these.
    all_src = df["src"].unique()
    src_with_clusters = df.loc[df["cluster"].notna()]["src"].unique()
    src_without_clusters = np.setdiff1d(all_src, src_with_clusters, assume_unique=True)
    df = df.loc[df["src"].isin(src_with_clusters)]
    logger.write("# seqs without clusters (removed these): " + str(len(src_without_clusters)))

    # Print no. of items that are not assigned to any cluster; remove these rows;
    # remove these.
    num_unassigned_items = df.groupby("src").agg({"cluster": lambda x: x.isna().sum()})
    num_items = df.groupby("src").agg({"cluster": len})
    num_unassigned_items.sort_values(by="src", inplace=True)
    num_items.sort_values(by="src", inplace=True)
    num_unassigned_items["cluster"] = num_unassigned_items["cluster"] / num_items["cluster"]
    p = [0, 50, 90, 99, 99.9, 100]
    logger.write(
        "percentage_unassigned_items_per_seq, percentiles, "
        + str(p)
        + str(np.percentile(num_unassigned_items["cluster"], p))
    )
    logger.write("avg: " + str(num_unassigned_items["cluster"].mean()))
    df = df.loc[df["cluster"].notna()]

    # save parsed data
    output_path = os.path.join(args.output_dir, "parsed_clusters.parquet")
    df.to_parquet(output_path, index=False)
    logger.write("Done")
