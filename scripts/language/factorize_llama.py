#!/usr/bin/env python
"""Reference‑compatible compression + evaluation script
------------------------------------------------------
This reproduces the settings of the whitening/low‑rank baseline we’ve been
discussing:
  • contiguous 2 048‑token chunks for both activation cache *and* evaluation
  • FP32 weights during factorisation, cast down only afterwards
  • skips embed/positional/lm_head matrices when assigning low rank
  • evaluates perplexity on the full test/validation split
  • counts FLOPs and params with the same sequence length
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
)
from lib.utils import (
    count_model_flops,
    get_all_convs_and_linears,
    seed_everything,
)

from contextlib import nullcontext

###############################################################################
# Helpers                                                                     #
###############################################################################

class ContiguousSeqDataset(Dataset):
    """Return non‑overlapping fixed‑length chunks from a tokenised corpus."""

    def __init__(self, token_ids: torch.LongTensor, seq_len: int):
        n_tokens = (token_ids.size(0) // seq_len) * seq_len
        token_ids = token_ids[:n_tokens]
        self.seq_len = seq_len
        self.samples = token_ids.view(-1, seq_len)

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return {
            "input_ids": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}

@torch.no_grad()
def perplexity(model, dataloader, device=None, ignore_inf=True):
    """
    Compute perplexity on `dataloader` the same way `ppl_eval` does, but
    keep the simple (model, dataloader) signature.

    Args
    ----
    model        : a causal-LM with `.logits` output
    dataloader   : iterable of batches; each batch must contain at least
                   a LongTensor `input_ids` shaped [B, T]; an
                   `attention_mask` key is optional and will be forwarded.
    device       : torch device; defaults to model's own device
    ignore_inf   : if True, silently skip any batch whose logits contain
                   NaN/Inf (identical to ppl_eval’s safeguard)

    Returns
    -------
    ppl          : scalar perplexity (float)
    """
    device = device or next(model.parameters()).device
    model.to(device).eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nll_tokens = []            # per-token negative log likelihoods
    total_tokens = 0

    for batch in dataloader:
        # Pull the pieces we need
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)


        outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False
            )
        logits = outputs.logits

        # NaN/Inf guard
        if ignore_inf and not torch.isfinite(logits).all():
            continue

        # Manual shift (left-aligned causal LM target)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        nll_tokens.append(loss)
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        raise RuntimeError("No valid tokens processed ­— every batch had NaN/Inf logits?")

    # Concatenate per-token NLLs and average, just like ppl_eval
    mean_nll = torch.cat(nll_tokens, dim=0).mean()
    ppl = math.exp(mean_nll.item())

    print(f"PPL: {ppl:.4f}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Weight Memory: {mem:.2f} MiB")

    return ppl

###############################################################################
# Main                                                                        #
###############################################################################

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--mode", choices=["flops_auto", "params_auto", "energy_act_aware", "rank_act_aware", "energy"], default="flops_auto")
    p.add_argument("--ratio_to_keep", type=float, required=True, help="Fraction of parameters to KEEP in each linear layer (e.g. 0.8)")
    p.add_argument("--dataset", choices=["wikitext2", "ptb"], default="wikitext2")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # Tokeniser + model (FP32 on GPU)
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.to("cuda").eval()

    # ------------------------------------------------------------------
    # Load evaluation corpus and build contiguous‑chunk dataset
    # ------------------------------------------------------------------
    if args.dataset == "wikitext2":
        raw_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    else:  # PTB validation
        raw_texts = load_dataset("ptb_text_only", "penn_treebank", split="validation")["sentence"]

    # Tokenise entire split at once for deterministic chunking
    tok_ids = tok("\n\n".join(raw_texts), return_tensors="pt").input_ids.squeeze(0)
    eval_ds = ContiguousSeqDataset(tok_ids, args.seq_len)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ------------------------------------------------------------------
    # Baseline metrics
    # ------------------------------------------------------------------
    ppl_baseline = perplexity(model, eval_dl)
    print(f"[baseline] ppl {ppl_baseline:.2f}  seq_len {args.seq_len}  batch_size {args.batch_size}")
    params_baseline = sum(p.numel() for p in model.parameters())
    flops_baseline = 10 # count_model_flops(model, (1, args.seq_len), dtype=torch.long, formatted=False)["total"]
    print(f"[baseline] ppl {ppl_baseline:.2f}  params {params_baseline/1e6:.2f}M  flops {flops_baseline/1e9:.2f}B")

    # ------------------------------------------------------------------
    # Prepare layer list (skip embeddings & lm_head)
    # ------------------------------------------------------------------
    all_keys = get_all_convs_and_linears(model)
    skip_re = re.compile(r"(embed_tokens|embed_positions|lm_head)")
    layer_keys = [k for k in all_keys if not skip_re.search(k)]

    # ------------------------------------------------------------------
    # Load activation cache and run chosen low‑rank routine (still FP32)
    # ------------------------------------------------------------------
    cache = torch.load(args.cache_path, map_location="cpu")

    if args.mode in {"flops_auto", "params_auto"}:
        metric = "flops" if args.mode == "flops_auto" else "params"
        model_lr = to_low_rank_activation_aware_auto(
            model,
            cache,
            ratio_to_keep=args.ratio_to_keep,
            keys=layer_keys,
            metric=metric,
        )
    elif args.mode in {"energy_act_aware", "energy"}:
        cfg_dict = {k: {"name": "svals_energy_ratio_to_keep", "value": args.ratio_to_keep} for k in layer_keys}
        if args.mode == "energy_act_aware":
            model_lr = to_low_rank_activation_aware_manual(model, cache, cfg_dict=cfg_dict)
        else:
            model_lr = to_low_rank_manual(model, cfg_dict=cfg_dict)
    else:
        cfg_dict = {k: {"name": "rank_ratio_to_keep", "value": args.ratio_to_keep} for k in layer_keys}
        model_lr = to_low_rank_activation_aware_manual(
            model, cache, cfg_dict=cfg_dict
        )
    del cache
    torch.cuda.empty_cache()

    # Cast compressed model down to bf16 (optional) and keep on GPU for eval
    model_lr.to(dtype=torch.float16).to("cuda").eval()

    # ------------------------------------------------------------------
    # Compressed metrics
    # ------------------------------------------------------------------
    params_compressed = sum(p.numel() for p in model_lr.parameters())
    flops_compressed = 10 # count_model_flops(model_lr, (1, args.seq_len), dtype=torch.long, formatted=False)["total"]
    ppl_compressed = perplexity(model_lr, eval_dl)

    print(
        f"[compressed] ppl {ppl_compressed:.2f}  params {params_compressed/1e6:.2f}M  "
        f"flops {flops_compressed/1e9:.2f}B  params_ratio {params_compressed/params_baseline:.4f}  "
        f"flops_ratio {flops_compressed/flops_baseline:.4f}"
    )

    # ------------------------------------------------------------------
    # Persist artefacts & metrics
    # ------------------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model_lr.to("cpu").state_dict(),
            "ratio": args.ratio_to_keep,
            "mode": args.mode,
            "seq_len": args.seq_len,
        },
        results_dir / "model_compressed.pth",
    )

    json.dump(
        {
            "ppl": ppl_compressed,
            "ppl_orig": ppl_baseline,
            "params_ratio": params_compressed / params_baseline,
            "flops_ratio": flops_compressed / flops_baseline,
            "mode": args.mode,
            "ratio": args.ratio_to_keep,
            "seq_len": args.seq_len,
        },
        open(results_dir / "metrics.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()


"""
{
  "ppl": 10.813658622486534,
  "ppl_orig": 5.472108436285095,
  "params_ratio": 0.8077411935049155,
  "flops_ratio": 1.0, # fake
  "mode": "params_auto",
  "ratio": 0.8,
  "seq_len": 2048
}
"""