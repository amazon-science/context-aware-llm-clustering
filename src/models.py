import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_sparset5 import SparseT5Stack
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)
from transformers import T5ForConditionalGeneration
from utils import Logger


def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
    no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write("# parameters: " + str(total))
    logger.write("# trainable parameters: " + str(trainable) + ", " + str(100 * trainable / total) + "%")

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write("#params by dtype:")
    for k, v in dtypes.items():
        logger.write(str(k) + ": " + str(v) + ", " + str(100 * v / total) + "%")


class ClusterLLM(nn.Module):
    """Model class for clustering and cluster name generation."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

        # load pretrained model
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")

        # remove decoder and spread encoder across all GPUs.
        self.model.lm_head = None
        device_map = self.model.hf_device_map
        num_enc_blocks = len(self.model.encoder.block)
        if args.gpus == "all":
            ngpus = torch.cuda.device_count()
            gpus = list(range(ngpus))
        else:
            gpus = list(map(int, args.gpus.split(",")))
            ngpus = len(gpus)
        self.first_device = torch.device("cuda:" + str(gpus[0]))
        num_blocks_per_gpu = num_enc_blocks // ngpus
        device_map["shared"] = gpus[0]
        device_map["encoder.embed_tokens"] = gpus[0]
        device_map["decoder.embed_tokens"] = gpus[0]
        for block_idx in range(num_enc_blocks):
            gpu_id = gpus[min(ngpus - 1, block_idx // num_blocks_per_gpu)]
            device_map["encoder.block." + str(block_idx)] = gpu_id
        device_map["encoder.final_layer_norm"] = gpu_id
        device_map["encoder.dropout"] = gpu_id
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map=device_map)
        self.model.decoder = None

        # change encoder to sparse encoder if needed
        if args.set_enc_type.startswith("sia"):
            self.model.encoder = self.create_sparse_encoder()

        # set loss function for clustering
        if "triplet" in args.loss_type:
            self.loss = self.triplet_loss
            if "neutral" in args.loss_type:
                self.neutral_sim = nn.Parameter(torch.tensor(args.cutoff), requires_grad=True)
        elif args.loss_type == "scl":
            self.loss = self.scl_loss
        else:
            self.loss = self.misc_loss

        # print no. of parameters
        count_parameters(args.logger, self.model)

    def create_sparse_encoder(self):
        # create sparse inter-item attention encoder
        sparse_encoder = SparseT5Stack(
            self.model.encoder.config, self.model.encoder.embed_tokens, self.args.set_enc_type
        )
        # load pretrained weights
        sparse_encoder.load_state_dict(self.model.encoder.state_dict())
        # partition sparse encoder across gpus
        for module_name, gpu_id in self.model.hf_device_map.items():
            device_str = "cuda:" + str(gpu_id)
            if module_name.startswith("encoder.block."):
                block_idx = int(module_name.split(".")[-1])
                sparse_encoder.block[block_idx] = sparse_encoder.block[block_idx].to(device_str)
            elif module_name == "encoder.final_layer_norm":
                sparse_encoder.final_layer_norm = sparse_encoder.final_layer_norm.to(device_str)
            elif module_name == "encoder.dropout":
                sparse_encoder.dropout = sparse_encoder.dropout.to(device_str)
            elif module_name == "encoder.embed_tokens":
                sparse_encoder.embed_tokens = sparse_encoder.embed_tokens.to(device_str)
        return sparse_encoder

    def scl_loss(self, pred, true, row_attention_mask):
        """
        Supervised clustering loss from
        https://assets.amazon.science/e0/01/95778ee44bc1bd5e0b7066899254/supervised-clustering-loss-for-clustering-friendly-sentence-embeddings-an-application-to-intent-clustering.pdf
        Following https://github.com/amazon-science/supervised-intent-clustering/blob/main/src/utils/losses.py
        pred -> bsz, max_items, max_items
        true -> bsz, max_items, max_items
        row_attention_mask -> bsz, max_items
        """
        device = pred.device
        pairwise_score_matrics = pred
        pairwise_class_equality = true
        pairwise_class_equality_negative = 1 - true

        C, r = self.args.C, self.args.r
        gold_similarity_matrix = -pairwise_score_matrics * pairwise_class_equality
        viol_similarity_matrix = (
            pairwise_score_matrics + pairwise_class_equality_negative * C * r - pairwise_class_equality * C
        )
        viol_similarity_matrix = -viol_similarity_matrix * (viol_similarity_matrix > 0).float()

        num_items = row_attention_mask.sum(dim=-1)  # bsz
        bsz = num_items.size()[0]
        viol_similarity_matrix = [viol_similarity_matrix[b, : num_items[b], : num_items[b]] for b in range(bsz)]
        gold_similarity_matrix = [gold_similarity_matrix[b, : num_items[b], : num_items[b]] for b in range(bsz)]
        pairwise_score_matrics = [pairwise_score_matrics[b, : num_items[b], : num_items[b]] for b in range(bsz)]
        pairwise_class_equality = [pairwise_class_equality[b, : num_items[b], : num_items[b]] for b in range(bsz)]
        pairwise_class_equality_negative = [
            pairwise_class_equality_negative[b, : num_items[b], : num_items[b]] for b in range(bsz)
        ]

        viol_spanning_tree = [minimum_spanning_tree(m.cpu().detach().numpy()).toarray() for m in viol_similarity_matrix]
        gold_spanning_tree = [minimum_spanning_tree(m.cpu().detach().numpy()).toarray() for m in gold_similarity_matrix]
        viol_spanning_tree = [torch.Tensor(m).to(device) for m in viol_spanning_tree]
        gold_spanning_tree = [torch.Tensor(m).to(device) for m in gold_spanning_tree]

        a = [torch.count_nonzero(m) for m in gold_spanning_tree]
        b = [torch.count_nonzero(m1 * m2) for m1, m2 in zip(viol_spanning_tree, pairwise_class_equality)]
        c = [torch.count_nonzero(m1 * m2) for m1, m2 in zip(viol_spanning_tree, pairwise_class_equality_negative)]
        delta = torch.tensor(a) - torch.tensor(b) + torch.tensor(c) * r
        delta = delta.to(device)

        loss_by_sample = []
        for b in range(bsz):
            viol_spanning_tree[b][viol_spanning_tree[b] != 0] = 1
            viol_score = torch.sum(pairwise_score_matrics[b] * viol_spanning_tree[b])
            gold_spanning_tree[b][gold_spanning_tree[b] != 0] = 1
            gold_score = torch.sum(pairwise_score_matrics[b] * gold_spanning_tree[b])
            obj = C * delta[b] + viol_score - gold_score
            loss = torch.max(torch.Tensor([0]).to(device), obj) * (delta[b] > 0).float()
            loss_by_sample.append(loss)
        return torch.stack(loss_by_sample).mean()

    def misc_loss(self, pred, true, row_attention_mask, col_attention_mask=None):
        """
        Computes one of double_margin, basic, cross_entropy losses.
        pred -> bsz, max_items, max_items (max_clusters)
        true -> bsz, max_items, max_items (max_clusters)
        row_attention_mask -> bsz, max_items
        col_attention_mask -> bsz, max_clusters
        """
        if self.args.loss_type == "basic":
            pos_loss, neg_loss = -pred, pred
        elif self.args.loss_type == "cross_entropy":
            pred = torch.clip((pred + 1) / 2, min=1e-5, max=1 - 1e-5)  # map [-1,1] to (0,1]
            pos_loss = -torch.log(pred)
            neg_loss = -torch.log(1 - pred)
        loss = true * pos_loss + (1 - true) * neg_loss
        if col_attention_mask is None:
            loss_mask = (
                row_attention_mask[:, :, None]
                * row_attention_mask[:, None, :]
                * (1 - torch.eye(pred.size()[1], device=loss.device))[None, :, :]
            )
        else:
            loss_mask = row_attention_mask[:, :, None] * col_attention_mask[:, None, :]
        loss = (loss * loss_mask).sum(dim=(1, 2)) / torch.clip(loss_mask.sum(dim=(1, 2)), min=1)
        return loss.mean()

    def triplet_loss(self, pred, true, row_attention_mask, col_attention_mask=None):
        """Computes triplet loss. Input shapes same as for misc_loss()."""
        loss = torch.clip(self.args.margin - pred[:, :, :, None] + pred[:, :, None, :], min=0)
        # bsz, max_items, max_items/clusters (p), max_items/clusters (n)
        if col_attention_mask is None:
            col_attention_mask = row_attention_mask
        loss_mask = (
            row_attention_mask[:, :, None, None]
            * col_attention_mask[:, None, :, None]
            * col_attention_mask[:, None, None, :]
        )
        loss_mask = loss_mask * true[:, :, :, None] * (1 - true)[:, :, None, :]
        numer = (loss * loss_mask).sum(dim=(1, 2, 3))
        denom = loss_mask.sum(dim=(1, 2, 3))

        if "neutral" in self.args.loss_type:
            # extra triplets with neutral node
            pos_loss = torch.clip(self.args.margin / 2 - pred + self.neutral_sim, min=0)
            neg_loss = torch.clip(self.args.margin / 2 - self.neutral_sim + pred, min=0)
            loss_mask = row_attention_mask[:, :, None] * col_attention_mask[:, None, :]
            numer_pos = (pos_loss * true * loss_mask).sum(dim=(1, 2))
            numer_neg = (neg_loss * (1 - true) * loss_mask).sum(dim=(1, 2))
            denom_pos = (true * loss_mask).sum(dim=(1, 2))
            denom_neg = ((1 - true) * loss_mask).sum(dim=(1, 2))
            pos_wt, neg_wt = 1, 1
            numer += pos_wt * numer_pos + neg_wt * numer_neg
            denom += pos_wt * denom_pos + neg_wt * denom_neg

        loss = numer / torch.clip(denom, min=1)
        return loss.mean()

    def forward(self, input_ids=None, attention_mask=None, item_item=None, item_end_pos=None, eval=False):
        """Forward function that calls the appropriate
        forward function for model type."""
        if self.args.set_enc_type == "nia" or self.args.set_enc_type.startswith("sia"):
            return self.forward_nia(input_ids, attention_mask, item_item, eval)
        elif self.args.set_enc_type == "fia":
            return self.forward_fia(input_ids, attention_mask, item_item, item_end_pos, eval)

    def normalize_emb(self, emb):
        """Return l2 normalized emb."""
        if self.args.loss_type not in ["log_sigmoid"]:
            emb = F.normalize(emb, p=2.0, dim=-1)
        return emb

    def forward_nia(self, input_ids, attention_mask, item_item, eval):
        """
        input_ids, attention_mask -> bsz, max_items, max_item_len
        item_item -> bsz, max_items, max_items
        """
        bsz, max_items, max_item_len = input_ids.size()
        if self.args.set_enc_type == "nia":
            encoder_outputs = self.model.encoder(
                input_ids=input_ids.reshape((-1, max_item_len)),
                attention_mask=attention_mask.reshape((-1, max_item_len)),
            )
            hid = encoder_outputs.last_hidden_state.reshape((bsz, max_items, max_item_len, -1))
        else:
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hid = encoder_outputs.last_hidden_state

        # item-item loss
        device = hid.device
        attention_mask = attention_mask.to(device)
        hid_mean = (hid * attention_mask[:, :, :, None]).sum(dim=-2) / torch.clip(
            attention_mask.sum(dim=-1, keepdims=True), min=1
        )
        item_emb = self.normalize_emb(hid_mean)

        if eval:
            return item_emb

        pred_item_item = torch.bmm(item_emb, item_emb.transpose(1, 2))
        # assuming right padding and taking item as padded iff first item token is pad
        loss = self.loss(pred_item_item, item_item.to(device), attention_mask[:, :, 0])

        return loss

    def forward_fia(self, input_ids, attention_mask, item_item, item_end_pos, eval):
        """
        input_ids, attention_mask -> bsz, max_inp_len
        item_item -> bsz, max_items, max_items
        """
        hid = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # -> bsz, max_inp_len, d
        item_end_pos = item_end_pos.to(hid.device)
        hid_mean = self.gather_item_emb(hid, item_end_pos)  # bsz, max_items, d
        item_emb = self.normalize_emb(hid_mean)

        if eval:
            return item_emb

        item_attention_mask = (item_end_pos > 0).int()
        pred_item_item = torch.bmm(item_emb, item_emb.transpose(1, 2))
        loss = self.loss(pred_item_item, item_item.to(hid.device), item_attention_mask)

        return loss

    def gather_item_emb(self, hid, item_end_pos):
        """
        Gather item/cluster embeddings by averaging the appropriate
        token embeddings.
        hid -> bsz, max_inp_len, d
        item_end_pos -> bsz, max_items
        """
        hid = torch.cat((torch.zeros_like(hid[:, :1, :]), hid), dim=1)
        item_end_pos = torch.cat((torch.zeros_like(item_end_pos[:, :1]), item_end_pos + 1), dim=1)
        hid = hid.cumsum(dim=1)  # bsz, max_inp_len, d
        item_end_pos = item_end_pos[:, :, None].repeat((1, 1, hid.size()[-1]))
        item_emb = torch.gather(hid, 1, item_end_pos)  # bsz, 1+max_items, d
        item_emb = item_emb[:, 1:, :] - item_emb[:, :-1, :]  # bsz, max_items,
        item_lens = item_end_pos[:, 1:, 0] - item_end_pos[:, :-1, 0]  # bsz, max_items
        item_emb = item_emb / torch.clip(item_lens[:, :, None], 1)
        return item_emb
