# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Griffin model."""
import pathlib

import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import torch
from torch import nn
from torch.utils import checkpoint

from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
from recurrentgemma.torch import layers
from recurrentgemma.torch import modules

weights_dir = pathlib.Path(kagglehub.model_download("google/recurrentgemma/pyTorch/2b-it"))
vocab = spm.SentencePieceProcessor()
vocab.Load(str(weights_dir / "tokenizer.model"))

Cache = dict[str, modules.ResidualBlockCache]


class Griffin(nn.Module):
    """Griffin model - https://arxiv.org/abs/2402.19427."""

    def __init__(
        self,
        config: common.GriffinConfig,
        gradient_checkpointing: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the Griffin model.

        Args:
          config: The Griffin config.
          gradient_checkpointing: Whether to apply gradient checkpointing on every
            residual block.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.embedder = modules.Embedder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.width,
            scale_by_sqrt_dim=self.config.embeddings_scale_by_sqrt_dim,
            device=device,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                modules.ResidualBlock(
                    width=self.config.width,
                    mlp_expanded_width=self.config.mlp_expanded_width,
                    num_heads=self.config.num_heads,
                    attention_window_size=self.config.attention_window_size,
                    temporal_block_type=block_type,
                    lru_width=self.config.lru_width,
                    final_w_init_variance_scale=2.0 / self.config.num_layers,
                    device=device,
                    dtype=dtype,
                )
                for block_type in self.config.block_types
            ]
        )
        self.final_norm = layers.RMSNorm(width=self.config.width, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.embedder.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.final_norm.reset_parameters()

    @at.typed
    def forward(
        self,
        tokens: at.Tokens,
        segment_pos: at.SegmentPos,
        cache: Cache | None = None,
        x_cache=None,
        decoded_toks_cache=None,
    ) -> tuple[at.TokenLogits, Cache]:
        """Calls Griffin.

        Args:
          tokens: Sequence of input tokens.
          segment_pos: Positions of each token in the sequence.
          cache: Optiona for the model.

        Returns:
          Output of the model together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        input_emb = self.embedder.encode(tokens)
        x = input_emb

        prompt_pass = True if x_cache is None else False
        x_cache = input_emb if x_cache is None else torch.cat([x_cache, x], dim=1)  # (b, L, e)
        toks_ = tokens[0].cpu().tolist()
        decoded_toks_ = [vocab.DecodeIds(_) for _ in toks_]
        decoded_toks_cache = decoded_toks_ if decoded_toks_cache is None else decoded_toks_cache + decoded_toks_

        all_layer_sims, all_layer_probs, yticklabels_rec, yticklabels_attn = [], [], [], []
        new_cache = {}
        for i, block in enumerate(self.blocks):
            block_name = f"blocks.{i}"
            block_cache = None if cache is None else cache[block_name]
            if self.gradient_checkpointing:
                x, new_cache[block_name] = checkpoint.checkpoint(
                    block,
                    x,
                    segment_pos,
                    block_cache,
                    decoded_toks_cache,
                    use_reentrant=False,
                    determinism_check="none",
                )
            else:
                x, new_cache[block_name] = block(x, segment_pos, block_cache, decoded_toks_cache)

            # print(block_name)
            # print(f"toks_={toks_}, len={len(toks_)}")
            # print(f"decoded_toks_={decoded_toks_}")
            # print(f"decoded_toks_cache={decoded_toks_cache}")
            # print(f"{x.shape=}")  # (b=1, L=1, e=2560)
            # print(f"{x_cache.shape=}")  # (b=1, L, e=2560)
            if block.temporal_block_type == common.TemporalBlockType.RECURRENT:
                # print(f"{new_cache[block_name].rg_lru_state.shape=}")  # latest hidden: (b=1, e=2560)
                sims = torch.nn.functional.cosine_similarity(
                    x_cache[0], new_cache[block_name].rg_lru_state[0], dim=1
                )  # (L,)
                # print(f"{sims.shape=}")
                all_layer_sims.append(sims.cpu().unsqueeze(0))  # (1, L)
                yticklabels_rec.append(i)
            else:
                attn_probs = new_cache[block_name].probs[0]  # (num_heads, L=1, L)
                attn_probs = torch.mean(attn_probs, dim=0)[:, 1:]  # (L=1, L-1)
                all_layer_probs.append(attn_probs.cpu().unsqueeze(0))
                yticklabels_attn.append(i)
            # logits = (
            #     nn.functional.tanh(self.embedder.decode(self.final_norm(x)) / self.config.logits_soft_cap)
            #     * self.config.logits_soft_cap
            # )
            # next_tok = torch.argmax(logits, axis=-1)[:, 0]
            # print(f"next_tok={next_tok.cpu().tolist()}, decoded_next_tok={vocab.DecodeIds(next_tok.cpu().tolist())}")
            # print("==BLOCK END==")

        x = self.final_norm(x)
        logits = self.embedder.decode(x)

        c = self.config.logits_soft_cap
        if c is not None:
            logits = nn.functional.tanh(logits / c) * c

        next_tok = torch.argmax(logits, axis=-1)[:, -1]
        # print(f"next_tok={next_tok.cpu().tolist()}, decoded_next_tok={vocab.DecodeIds(next_tok.cpu().tolist())}")
        print(f"next_tok={vocab.DecodeIds(next_tok.cpu().tolist())}")

        print("recurrence =>")
        plt.figure(figsize=(8, 5))
        all_layer_sims = torch.cat(all_layer_sims, dim=0).float()  # (num_recurrent_blocks, L)
        sns.heatmap(all_layer_sims.numpy(), xticklabels=decoded_toks_cache, yticklabels=yticklabels_rec)
        plt.xticks(rotation=90)
        plt.show()

        plt.figure(figsize=(8, 0.4))
        sns.heatmap(torch.mean(all_layer_sims, dim=0).unsqueeze(0), xticklabels=decoded_toks_cache, yticklabels=["  "])
        plt.xticks(rotation=90)
        plt.show()

        print("attn =>")
        all_layer_probs = torch.cat(all_layer_probs, dim=0).float()  # (num_attn_blocks, 1, L-1)
        if not prompt_pass:
            plt.figure(figsize=(8, 5))
            sns.heatmap(
                all_layer_probs.squeeze(1).numpy(), xticklabels=decoded_toks_cache[1:], yticklabels=yticklabels_attn
            )
            plt.xticks(rotation=90)
            plt.show()

        yticklabels = decoded_toks_cache if prompt_pass else [decoded_toks_cache[-1]]
        if prompt_pass:
            plt.figure(figsize=(8, 5))
        else:
            plt.figure(figsize=(8, 0.4))
        sns.heatmap(torch.mean(all_layer_probs, dim=0), xticklabels=decoded_toks_cache[1:], yticklabels=yticklabels)
        plt.xticks(rotation=90)
        plt.show()



        print("\n==END==\n")

        return logits, new_cache, x_cache, decoded_toks_cache

    def init_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
    ) -> Cache:
        """Initializes an empty cache for the model."""
        cache = {}
        for i, block_type in enumerate(self.config.block_types):
            cache[f"blocks.{i}"] = modules.ResidualBlock.init_cache(
                batch_size=batch_size,
                width=self.config.width,
                num_heads=self.config.num_heads,
                attention_window_size=self.config.attention_window_size,
                temporal_block_type=block_type,
                dtype=dtype,
                lru_width=self.config.lru_width,
            )
        return cache
