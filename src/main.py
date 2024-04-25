"""Run this to train/evaluate clustering model."""

import argparse
import os

import numpy as np
import torch
from dataset import Dataset
from evaluate_cluster import Evaluator
from models import ClusterLLM
from pretrain_dataset import PretrainDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
from utils import Logger, save_ckpt, set_all_seeds


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--gpus", default="all", type=str)

    # dataset related arguments
    parser.add_argument("--dataset", type=str, default="q2q_gifts")
    parser.add_argument("--max_tokens_per_text", type=int, default=32)
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="no. of training item sets if integer>1," "otherwise fraction of item sets to use for training.",
    )
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_clusters", type=int, default=None)

    # model related arguments
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--set_enc_type", type=str, default="nia", choices=["nia", "fia", "sia_first", "sia_hid_mean"])
    parser.add_argument("--load_ckpt_path", type=str, default=None)

    # loss related arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="triplet_neutral",
        choices=["basic", "cross_entropy", "scl", "triplet", "triplet_neutral"],
    )
    parser.add_argument("--margin", type=float, default=0.3, help="margin foir cfd and triplet losses")
    parser.add_argument("--cutoff", type=float, default=0.0, help="merge cutoff for cfd loss")
    parser.add_argument("--C", type=float, default=0.15, help="param for scl loss")
    parser.add_argument("--r", type=float, default=0.5, help="param for scl loss")
    parser.add_argument("--tau", type=float, default=0.5, help="param for ntxent loss")

    # training/eval realated arguments
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_dir_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--print_train_loss_every", type=int, default=100)
    parser.add_argument("--validate_after", type=int, default=0)
    parser.add_argument("--validate_every", type=int, default=None)

    args = parser.parse_args()
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir
    if it is not passed in args."""
    if args.output_dir is None:
        if args.pretrain:
            args.output_dir_prefix = "pretrain_" + args.output_dir_prefix
        elif args.load_ckpt_path is not None:
            args.output_dir_prefix = "finetune_" + args.output_dir_prefix
        args.output_dir = "../outputs/" + args.dataset + "/" + args.output_dir_prefix
        for argument in [
            "set_enc_type",
            "loss_type",
            "margin",
            "cutoff",
            "C",
            "r",
            "tau",
            "max_items",
            "max_clusters",
            "train_size",
            "model_name",
        ]:
            args.output_dir += argument + ":" + str(getattr(args, argument)).replace("/", "-") + "|"
        args.output_dir = args.output_dir[:-1]
        # reduce path length
        for s in ["set_enc_type", "loss_type"]:
            args.output_dir = args.output_dir.replace(s + ":", "")
    os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, "log.txt")
    args.logger.write("\n" + str(args))
    args.device = torch.device("cuda")
    set_all_seeds(args.seed)
    model_path_best = os.path.join(args.output_dir, "clus_checkpoint_best.bin")

    # load model and tokenizer
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ClusterLLM(args)
    if args.load_ckpt_path is not None:
        state_dict = torch.load(args.load_ckpt_path)["model_weights"]
        for k in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]:
            state_dict[k] = state_dict["model.shared.weight"]
        model.load_state_dict(state_dict)

    # load data
    dataset = PretrainDataset(args) if args.pretrain == 1 else Dataset(args)

    # training loop
    num_train = len(dataset.splits["train"])
    num_batches_per_epoch = num_train / args.train_batch_size
    args.logger.write("\nNo. of training batches per epoch = " + str(num_batches_per_epoch))
    args.max_steps = int(round(num_batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(num_batches_per_epoch)
    cum_train_loss, num_steps, num_batches_trained = 0, 0, 0
    wait, patience_reached = args.patience, False
    best_val_metric = -np.inf
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    train_bar = tqdm(range(args.max_steps))
    evaluator = Evaluator(args)

    # results before any training
    if args.validate_after < 0:
        results = evaluator.evaluate(model, dataset, "val", train_step=-1)
        if args.pretrain != 1:
            evaluator.evaluate(model, dataset, "eval_train", train_step=-1, best_cutoff=results["best_cutoff"])
            evaluator.evaluate(model, dataset, "test", train_step=-1, best_cutoff=results["best_cutoff"])

    model.train()
    for step in train_bar:
        # load batch
        batch = dataset.get_batch()
        batch = {k: v.cuda() for k, v in batch.items()}

        # forward pass
        loss = model(**batch)

        # backward pass
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # add to cum loss
        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        # Log training losses.
        train_bar.set_description(str(np.round(cum_train_loss / num_batches_trained, 5)))
        if (num_steps) % args.print_train_loss_every == 0:
            args.logger.write(
                "\nTrain-loss at step " + str(num_steps) + ": " + str(cum_train_loss / num_batches_trained)
            )
            cum_train_loss, num_batches_trained = 0, 0

        # run validatation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            # get metrics on test and validation splits
            results = evaluator.evaluate(model, dataset, "val", train_step=step)
            if args.pretrain != 1:
                evaluator.evaluate(model, dataset, "eval_train", train_step=step, best_cutoff=results["best_cutoff"])
                test_res = evaluator.evaluate(
                    model, dataset, "test", train_step=step, best_cutoff=results["best_cutoff"]
                )
            model.train(True)

            # Save ckpt if there is an improvement.
            curr_val_metric = sum([results[results["best_cutoff"]][m][0] for m in ["NMI", "AMI", "RI", "ARI"]])
            if curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                args.logger.write("\nSaving ckpt at " + model_path_best)
                save_ckpt(model_path_best, model, optimizer, num_steps, best_val_metric)
                if args.pretrain != 1:
                    best_test_res = test_res
            # print test res with best val res and best val metric
            if args.pretrain != 1:
                args.logger.write("\nTest res with best val ckpt: " + str(best_test_res))
            args.logger.write("\nBest val metric: " + str(best_val_metric))
