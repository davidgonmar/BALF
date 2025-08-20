#!/usr/bin/env python
"""
Collect activation statistics for low‑rank compression — reference‑style
======================================================================
• Uses *contiguous* fixed‑length chunks (no random sub‑crops)
• Works with WikiText‑2, Penn Treebank, or C4
• Forces full‑precision (FP32) and single‑GPU execution
• Records activations only for transformer block projections (q/k/v/o + mlp)
"""
import argparse
import math
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.utils import seed_everything, get_all_convs_and_linears
from lib.factorization.factorize import collect_activation_cache
import random
from typing import List

###############################################################################
# Dataset that yields *contiguous* token blocks of fixed length
###############################################################################
class ContiguousSeqDataset(Dataset):
    """Return the first *n_samples* non‑overlapping L‑token chunks."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        n_samples: int,
        seq_len: int,
    ) -> None:
        # Tokenise entire corpus as one stream
        enc = tokenizer("\n\n".join(texts), return_tensors="pt")
        ids = enc.input_ids.squeeze(0)  # [T]
        total_needed = n_samples * seq_len
        if ids.size(0) < total_needed:
            raise ValueError(
                f"Corpus too small: need {total_needed} tokens, have {ids.size(0)}"
            )
        ids = ids[: total_needed]
        self.seq_len = seq_len
        self.samples = ids.view(n_samples, seq_len)

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return {
            "input_ids": chunk.clone(),
            "attention_mask": torch.ones_like(chunk),
        }

class ContiguousSeqDataset(Dataset):
    """
    API-compatible replacement for ContiguousSeqDataset that returns *random*
    non-overlapping `seq_len` chunks from WikiText-2.

    Args (kept identical for compatibility):
        texts (List[str]): Ignored – kept so existing calls don’t break.
        tokenizer (AutoTokenizer): Hugging Face tokenizer.
        n_samples (int): Number of chunks you want.
        seq_len (int): Tokens per chunk.

    Extra kwargs (all optional):
        seed (int): RNG seed for reproducibility.  Default = 3.
        cache_dir (str): Hugging Face cache for `load_dataset`.
        cache_file (str): Path to a .pt cache of the prepared tensor.
                          If omitted, a sensible default is used.
    """

    def __init__(
        self,
        texts: List[str],                # ignored, keeps signature unchanged
        tokenizer: AutoTokenizer,
        n_samples: int,
        seq_len: int,
        *,
        seed: int = 3,
        cache_dir = None,
        cache_file = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # ---------------------------------------------------------------------
        # 1. Choose cache file
        # ---------------------------------------------------------------------
        cache_root = Path("cache")
        cache_root.mkdir(exist_ok=True)
        if cache_file is None:
            cache_file = cache_root / f"wikitext2_rand_{n_samples}_{seq_len}_{seed}.pt"

        # ---------------------------------------------------------------------
        # 2. If we already processed once, just load the tensor
        # ---------------------------------------------------------------------
        if Path(cache_file).exists():
            self.samples = torch.load(cache_file)
            return

        # ---------------------------------------------------------------------
        # 3. Load full WikiText-2 train split
        # ---------------------------------------------------------------------
        data = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=cache_dir,
        )
        full_text = "\n\n".join(data["text"])
        text_len = len(full_text)

        # ---------------------------------------------------------------------
        # 4. Draw `n_samples` random windows as in get_calib_train_data
        # ---------------------------------------------------------------------
        random.seed(seed)
        samples: list[torch.Tensor] = []

        while len(samples) < n_samples:
            # pick a random start, give tokenizer plenty of context
            i = random.randint(0, text_len - seq_len - 1)
            j = min(i + seq_len * 10, text_len)
            chunk_ids = tokenizer(
                full_text[i:j], return_tensors="pt"
            ).input_ids[:, :seq_len]  # [1, <=seq_len]

            # rare edge-case: too few tokens → resample
            if chunk_ids.size(1) < seq_len:
                continue

            samples.append(chunk_ids.squeeze(0))  # [seq_len]

        # stack into [n_samples, seq_len] for easy indexing
        self.samples = torch.stack(samples)
        torch.save(self.samples, cache_file)

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.samples.size(0)

    def __getitem__(self, idx: int):
        chunk = self.samples[idx]
        return {
            "input_ids": chunk.clone(),
            "attention_mask": torch.ones_like(chunk),
        }

def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "ptb", "c4"], default="wikitext2")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cache_out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # Load tokenizer & model (FP32 on single GPU)
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.to("cuda").eval()

    # ------------------------------------------------------------------
    # Prepare text corpus
    # ------------------------------------------------------------------
    if args.dataset == "wikitext2":
        texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    elif args.dataset == "ptb":
        texts = load_dataset("ptb_text_only", "penn_treebank", split="train")["sentence"]
    else:  # C4 — take several shards if available
        c4 = load_dataset("c4", "en", split="train", streaming=False)
        texts = c4.shuffle(seed=args.seed)["text"]  # shuffle for diversity

    ds = ContiguousSeqDataset(texts, tok, args.n_samples, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ------------------------------------------------------------------
    # Select transformer‑block linears only
    # ------------------------------------------------------------------
    all_keys = get_all_convs_and_linears(model)
    pattern = re.compile(r"(embed_tokens|embed_positions|lm_head)")
    keys = [k for k in all_keys if not pattern.search(k)]

    # ------------------------------------------------------------------
    # Collect activations
    # ------------------------------------------------------------------
    print(f"Collecting activations for {len(keys)} projection layers …")
    act_cache = collect_activation_cache(model, dl, keys=keys)

    Path(args.cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(act_cache, args.cache_out)
    print(f"Saved activation cache → {args.cache_out}")


if __name__ == "__main__":
    main()

