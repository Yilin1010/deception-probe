"""CPU-friendly activation extractor for the live notebook demo.

The committed analysis.ipynb runs this on `gpt2` (124M, ~10s on CPU) over a
tiny REPE+DBench subset to prove the pipeline works end-to-end without GPU /
quantization. The Llama-3.3-70B activation extraction used in the blog ran on
an H200 with the same prompt/response alignment logic.

Output format matches the .npz schema used in the blog pipeline:
  activations.npy        (N_tok, n_layers_requested, hidden)  fp16
  sample_idx.npy         (N_tok,)                              int32
  token_positions.npy    (N_tok,)                              int32
  layer_indices.npy      (n_layers_requested,)                 int32  1-based
  sample_meta_json.npy   ()                                    object  JSON

Usage:
  python -m src.extract_activations \\
      --model gpt2 \\
      --layers 3,6,9 \\
      --in /tmp/toy.jsonl \\
      --out /tmp/toy.npz

Input JSONL: one {"prompt": ..., "response": ..., "label": 0|1, "sample_id": ...,
              "pair_id"?: int} per line.
"""
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _bpe_safe_response_ids(tokenizer, prompt: str, response: str):
    """Tokenize prompt and prompt+response; return (full_ids, prompt_len) where
    prompt_len is the longest common prefix length.

    Why this is non-trivial:

    To extract activations only at response tokens, we need to know where the
    response starts in the joint tokenization of prompt + response. The naive
    approach -- `full_ids[len(tokenize(prompt)):]` -- is wrong because Llama's
    BPE tokenizer is greedy on the *full string*, not on prompt and response
    separately. The last char of the prompt can merge with the first char of
    the response into a single token, so the prompt and response token spans
    overlap by one position.

    Failure mode is silent: mean-aggregated AUROC looks fine because one
    boundary token barely shifts the average, but per-token traces are off by
    one (a spike attributed to "fake number" actually lands on the previous
    token) and last-token aggregation can flip the sample.

    Fix: longest-common-prefix matching. Walk `prompt_ids` and `full_ids`
    together; the first index where they diverge is the true response start.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=True)["input_ids"]
    p_len = len(prompt_ids)
    if full_ids[:p_len] != prompt_ids:
        p_len = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            p_len += 1
    return full_ids, p_len


@torch.no_grad()
def extract(model_name: str, samples: List[dict], layers_1based: List[int],
            out_path: str) -> None:
    print(f"[extract] loading {model_name} on CPU...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    for L in layers_1based:
        if not 1 <= L <= n_layers:
            raise ValueError(f"layer {L} out of range [1, {n_layers}]")
    layer_idx_0 = [L - 1 for L in layers_1based]   # output_hidden_states[1] = layer 1

    flat_acts: List[np.ndarray] = []
    flat_sample_idx: List[int] = []
    flat_token_pos: List[int] = []
    meta_out: List[dict] = []

    for s_i, s in enumerate(samples):
        prompt, response = s["prompt"], s["response"]
        full_ids, p_len = _bpe_safe_response_ids(tok, prompt, response)
        if len(full_ids) <= p_len:
            continue   # empty response
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # output.hidden_states is tuple of length n_layers+1; index 0 = embeddings.
        # Stack only the layers we want: result shape (1, seq, n_layers_kept, hidden)
        hs = torch.stack([out.hidden_states[L] for L in layers_1based], dim=2)
        # Keep response tokens only.
        resp_hs = hs[0, p_len:].to(torch.float16).numpy()  # (T, n_layers_kept, hidden)
        T = resp_hs.shape[0]
        flat_acts.append(resp_hs)
        flat_sample_idx.extend([s_i] * T)
        flat_token_pos.extend(range(T))
        token_texts = [tok.decode([t]) for t in full_ids[p_len:]]
        meta_out.append({
            "sample_id": s.get("sample_id", f"sample_{s_i}"),
            "pair_id": s.get("pair_id"),
            "label": s.get("label"),
            "domain": s.get("domain"),
            "prompt": prompt,
            "response": response,
            "response_token_text": token_texts,
            "n_response_tokens_kept": T,
        })

    if not flat_acts:
        raise RuntimeError("no samples produced any tokens")

    activations = np.concatenate(flat_acts, axis=0)   # (sum_T, n_layers_kept, hidden)
    sample_idx = np.array(flat_sample_idx, dtype=np.int32)
    token_positions = np.array(flat_token_pos, dtype=np.int32)
    layer_indices = np.array(layers_1based, dtype=np.int32)
    sample_meta = np.array(json.dumps(meta_out), dtype=object)

    np.savez(
        out_path,
        activations=activations,
        sample_idx=sample_idx,
        token_positions=token_positions,
        layer_indices=layer_indices,
        sample_meta_json=sample_meta,
    )
    print(f"[extract] saved {out_path}: activations={activations.shape} "
          f"({activations.nbytes / 1e6:.1f} MB), {len(meta_out)} samples", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", required=True,
                    help="comma-separated 1-based layer indices, e.g. '3,6,9'")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="JSONL: one {prompt, response, label, sample_id, ...} per line")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    samples = []
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    extract(args.model, samples, layers, args.out)


if __name__ == "__main__":
    main()
