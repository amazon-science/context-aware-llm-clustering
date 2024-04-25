"""Run this code to get clusterings from LLM_c for Amazon review datasets."""

import argparse
import json
import os
import time
from types import NoneType
from typing import List, Tuple, Union

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from tqdm import tqdm
from utils import Logger, get_curr_time


def parse_args() -> argparse.Namespace:
    """Function for parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="name of the bedrock model to use")
    parser.add_argument("--dataset", type=str, help="name of the dataset")
    parser.add_argument("--max_item_len", type=int, default=200, help="max length of an item in characters")
    parser.add_argument("--max_seq_len", type=int, default=18, help="max length of a sequence in items")
    parser.add_argument("--max_response_len", type=int, default=5000, help="max response length in tokens")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="dir to store outputs, automatically set if not provided"
    )
    parser.add_argument("--max_tries", type=int, default=5, help="no. of times to try when prompting fails")
    return parser.parse_args()


def set_output_dir(args: argparse.Namespace) -> None:
    """Function to set output dir if None is passed."""
    if args.output_dir is None:
        args.output_dir = "../outputs/" + args.dataset + "/" + args.model_name.replace("/", "-")
    os.makedirs(args.output_dir, exist_ok=True)


class PromptGen:
    """Class to generate prompts from a fixed template."""

    def __init__(self, args: argparse.Namespace) -> None:
        # <SEQ> can be substituted later with desired sequnce
        self.template = "Human: Cluster these products:\n<SEQ>"
        self.template += (
            "\n\nFor each cluster, the answer should contain a meaningful "
            "cluster title and the products in that cluster. Do not "
            "provide any explanation."
        )
        self.template += "\n\nAssistant:"
        self.max_item_len = args.max_item_len
        self.max_seq_len = args.max_seq_len

    def generate_prompt(self, seq: List[str]) -> str:
        "Function to generate prompt for one sequence."
        # Trim seq according to max seq len. Trim each item according to max item len.
        seq = [item[: self.max_item_len] for item in seq[-self.max_seq_len :]]
        return self.template.replace("<SEQ>", "\n".join(seq))


def read_train_seqs(dataset: str) -> Tuple[dict, dict]:
    """Function to read train sequences from the dataset."""
    data_dir = "../data/" + dataset + "/"
    with open(data_dir + "train.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(data_dir + "meta_data.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(data_dir + "smap.json", encoding="utf-8") as f:
        smap = json.load(f)
    meta = {smap[k]: v for k, v in meta.items() if k in smap}
    return train, meta


def load_data(args: argparse.Namespace) -> dict:
    """Function to read train sequences from the dataset
    and generate prompts for them."""
    prompts = {}
    generate_prompt = PromptGen(args).generate_prompt

    # Read data
    train, meta = read_train_seqs(args.dataset)
    # Generate prompts
    for seq_id, seq in train.items():
        prompts[seq_id + "-train"] = generate_prompt([meta[item_id]["title"] for item_id in seq])
    # Print examples
    args.logger.write("\nExamples:")
    count = 0
    for prompt in prompts.values():
        args.logger.write("\n" + "*" * 20)
        args.logger.write("\nPrompt : " + prompt)
        count += 1
        if count == 5:
            break
    return prompts


class BedRock:
    """Class to intialize bedrock client, prompt and gather responses."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.bedrock = boto3.client("bedrock-runtime", "us-east-1")
        self.model_id = args.model_name
        self.accept = "*/*"
        self.content_type = "application/json"
        self.max_response_len = args.max_response_len
        self.max_tries = args.max_tries
        self.logger = args.logger

    def get_reponse(self, prompt: str, tries: int = 1) -> Union[str, NoneType]:
        """Prompt bedrock model and get response."""
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens_to_sample": self.max_response_len,
                "temperature": 1,
                "top_k": 1,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],
            }
        )
        try:
            response = self.bedrock.invoke_model(
                body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
            )
            response = json.loads(response.get("body").read())["completion"]
            return response
        except ClientError as exception_obj:
            if (exception_obj.response["Error"]["Code"] == "ThrottlingException") and (tries < self.max_tries):
                self.logger.write("Trying again - attempt no. " + str(tries + 1))
                time.sleep(10)
                return self.get_reponse(prompt, tries + 1)
            else:
                self.print_error(exception_obj, prompt)
                return None
        except Exception as exception_obj:
            self.print_error(exception_obj, prompt)
            return None

    def print_error(self, exception_obj: Exception, prompt: str) -> None:
        "Print error message along with the prompt that caused the error."
        self.logger.write("\nError while prompting: " + str(exception_obj))
        self.logger.write("Prompt: " + prompt)


if __name__ == "__main__":
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, "log.txt")
    args.logger.write("\n" + str(args))

    # load data and construct prompts
    prompts = load_data(args)

    # setup bedrock for prompting
    bedrock = BedRock(args)

    # create responses file with header if it doesn't exist
    responses_path = os.path.join(args.output_dir, "responses.csv")
    if not os.path.exists(responses_path):
        with open(responses_path, "w") as f:
            f.write("timestamp,seq_id,prompt,response\n")
        seen_seq_ids = set()
    else:
        # if responses file already exists, remove prompts that
        # have responses already
        seen_seq_ids = set(pd.read_csv(responses_path, usecols=["seq_id"])["seq_id"])

    # pass each prompt and get response
    for seq_id, prompt in tqdm(prompts.items(), total=len(prompts)):
        if seq_id in seen_seq_ids:
            continue
        response = bedrock.get_reponse(prompt)

        # print response and write it to file
        print("*" * 100)
        print("Prompt: ", prompt)
        print("Response: ", response)
        with open(responses_path, "a") as f:
            line = [get_curr_time(), seq_id, prompt, response]
            line = ",".join([str(cell).replace(",", "<COMMA>").replace("\n", "<NEWLINE>") for cell in line]) + "\n"
            try:
                f.write(line)
            except Exception:
                args.logger.write("Could not write this line to responses_path: " + line)
