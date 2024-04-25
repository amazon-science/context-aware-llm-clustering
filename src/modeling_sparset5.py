import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5LayerSelfAttention,
    T5Stack,
)


class SparseT5Attention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False, type="sia"):
        super().__init__(config, has_relative_attention_bias)
        self.type = type

    def compute_bias(self, max_item_len, max_items, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(max_item_len, dtype=torch.long, device=device)[:, None]
        memory_position_intra = torch.arange(max_item_len, dtype=torch.long, device=device)
        memory_position_inter = torch.ones(max_items, dtype=torch.long, device=device) * (
            max_item_len + self.relative_attention_max_distance + 5
        )
        memory_position = torch.cat((memory_position_intra, memory_position_inter))[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = (
            values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        )  # shape (1, 1, num_heads, query_length, key_length)
        return values

    def compute_bias_wo_pi(self, mask):
        # mask -> bsz, max_items, max_item_len
        bsz, max_items, max_item_len = mask.size()
        device = mask.device

        context_position = mask.cumsum(dim=-1) - 1  # bsz, max_items, max_item_len
        memory_position_intra = torch.clone(context_position)  # bsz, max_items, max_item_len
        memory_position_inter = torch.arange(1, max_items + 1, dtype=torch.long, device=device)[
            None, None, :
        ]  # 1, 1, max_items
        memory_position_inter = memory_position_inter + memory_position_intra[:, :, -1:]
        memory_position = torch.cat(
            (memory_position_intra, memory_position_inter), dim=-1
        )  # bsz, max_items, max_item_len+max_items
        relative_position = memory_position.unsqueeze(-2) - context_position.unsqueeze(-1)
        # -> bsz, max_items, max_item_len, max_item_len+max_items
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # -> bsz, max_items, max_item_len, max_item_len+max_items, num_heads
        values = values.permute([0, 1, 4, 2, 3])
        # -> bsz, max_items, num_heads, max_item_len, max_item_len+max_items
        return values

    def forward(self, hidden_states, mask, position_bias=None):
        """
        hidden_states -> bsz, max_items, max_item_len, d
        mask -> bsz, max_items, max_item_len
        position_bias -> bsz, max_items, n_heads, max_item_len, max_item_len+max_item_len
        """
        bsz, max_items, max_item_len, d = hidden_states.size()

        query_states = (
            self.q(hidden_states)
            .view((bsz, max_items, max_item_len, self.n_heads, self.key_value_proj_dim))
            .transpose(2, 3)
        )
        key_states = (
            self.k(hidden_states)
            .view((bsz, max_items, max_item_len, self.n_heads, self.key_value_proj_dim))
            .transpose(2, 3)
        )
        value_states = (
            self.v(hidden_states)
            .view((bsz, max_items, max_item_len, self.n_heads, self.key_value_proj_dim))
            .transpose(2, 3)
        )
        # -> bsz, max_items, n_heads, max_item_len, dk

        if self.type.startswith("sia_first"):
            # first_key_states = key_states[:,:,:,0,:]
            # first_value_states = value_states[:,:,:,0,:]
            mask2 = (torch.arange(max_item_len, device=hidden_states.device) == 0).int().reshape((1, 1, 1, -1, 1))
            first_key_states = (key_states * mask2).sum(dim=-2)
            first_value_states = (value_states * mask2).sum(dim=-2)
        elif self.type.startswith("sia_mean"):
            mask2 = mask[:, :, None, :, None]  # bsz, max_items, 1, max_item_len, 1
            item_lens = torch.clip(mask2.sum(dim=-2), min=1)
            first_key_states = (key_states * mask2).sum(dim=-2) / item_lens
            first_value_states = (value_states * mask2).sum(dim=-2) / item_lens
        elif self.type.startswith("sia_hid_mean"):
            mask2 = mask[:, :, :, None]
            hid_mean = (hidden_states * mask2).sum(dim=-2) / torch.clip(mask2.sum(dim=-2), min=1)
            # -> bsz, max_items, d
            first_key_states = self.q(hid_mean).view((bsz, max_items, self.n_heads, self.key_value_proj_dim))
            first_value_states = self.v(hid_mean).view((bsz, max_items, self.n_heads, self.key_value_proj_dim))

        first_key_states = first_key_states.transpose(1, 2).unsqueeze(1)
        first_value_states = first_value_states.transpose(1, 2).unsqueeze(1)
        # -> bsz, 1, n_heads, max_items, dk

        key_states = torch.cat((key_states, first_key_states.repeat((1, max_items, 1, 1, 1))), dim=-2)
        value_states = torch.cat((value_states, first_value_states.repeat((1, max_items, 1, 1, 1))), dim=-2)
        # -> bsz, max_items, n_heads, max_item_len+max_items, dk

        scores = torch.matmul(query_states, key_states.transpose(3, 4))
        # -> bsz, max_items, n_heads, max_item_len, max_item_len+max_items

        # POSITION-BIAS & PADDING
        if position_bias is None:
            mask = mask.to(scores.device)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, max_item_len, max_item_len + max_item_len),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            elif "wo_pi" in self.type:
                position_bias = self.compute_bias_wo_pi(mask)
            else:
                position_bias = self.compute_bias(max_item_len, max_items, device=scores.device)

            if mask is not None:
                mask_right = mask[:, :, 0]  # bsz, max_items
                mask_right = (
                    mask_right[:, :, None]
                    * mask_right[:, None, :]
                    * (1 - torch.eye(max_items, device=scores.device))[None, :, :]
                )
                # -> bsz, max_items, max_items
                mask = torch.cat((mask, mask_right), dim=2)
                # -> bsz, max_items, max_item_len+max_items
                mask = mask.to(dtype=scores.dtype)  # fp16 compatibility
                mask = (1.0 - mask) * torch.finfo(scores.dtype).min
                position_bias = position_bias + mask[:, :, None, None, :]
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        # -> bsz, max_items, n_heads, max_item_len, dk
        attn_output = attn_output.transpose(2, 3).contiguous().view(bsz, max_items, -1, self.inner_dim)
        # -> bsz, max_items, max_item_len, d
        attn_output = self.o(attn_output)

        return attn_output, position_bias, scores


class SparseT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False, type="sia"):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = SparseT5Attention(config, has_relative_attention_bias, type)

    def forward(self, hidden_states, attention_mask, position_bias=None):
        device = self.layer_norm.weight.device
        hidden_states = hidden_states.to(device)
        attention_mask = attention_mask.to(device)
        if position_bias is not None:
            position_bias = position_bias.to(device)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return hidden_states, attention_output[1], attention_output[2]


class SparseT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, type="sia"):
        super().__init__(config, has_relative_attention_bias)
        self.layer[0] = SparseT5LayerSelfAttention(config, has_relative_attention_bias, type)

    def forward(self, hidden_states, attention_mask, position_bias=None):

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states, position_bias, scores = self_attention_outputs

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, position_bias, scores


class SparseT5Stack(T5Stack):
    """Class to modify T5Stack for item set clustering
    with sparse interaction between items."""

    def __init__(self, config, embed_tokens=None, type="sia"):
        super().__init__(config, embed_tokens)
        self.block = nn.ModuleList(
            [
                SparseT5Block(config, has_relative_attention_bias=bool(i == 0), type=type)
                for i in range(config.num_layers)
            ]
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        input_ids, attention_mask -> bsz, max_items, max_item_len
        """

        position_bias = None
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        all_hidden_states = ()
        attention_scores = ()

        for i, layer_module in enumerate(self.block):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )
            hidden_states, position_bias, scores = layer_outputs
            attention_scores = attention_scores + (scores,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=attention_scores
        )
