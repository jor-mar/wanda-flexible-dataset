import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer

from main_opt import get_llm
from lib.prune_opt import prune_wanda, check_sparsity
from lib.eval import eval_ppl


def save_original_weights(model):
    original_weights = {}

    for name, param in model.named_parameters():
        original_weights[name] = param.detach().cpu().clone()

    return original_weights


def restore_original_weights(model, original_weights):
    for name, param in model.named_parameters():
        param.data.copy_(original_weights[name].to(param.device))


def run_experiment(
    model,
    tokenizer,
    device,
    original_weights,
    args,
    dataset,
    seed
):

    print("\n" + "=" * 70)
    print(f"STARTING | {dataset} | SEED {seed}")
    print("=" * 70)

    # Restore the original unpruned model.
    restore_original_weights(model, original_weights)

    # Set random seeds.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set the calibration dataset and seed for this run.
    args.calib_dataset = dataset
    args.seed = seed

    # Run Wanda.
    prune_wanda(
        args,
        model,
        tokenizer,
        device,
        prune_n=0,
        prune_m=0
    )

    # Check actual sparsity.
    sparsity = check_sparsity(model)

    # Evaluate on WikiText-2.
    ppl = eval_ppl(args, model, tokenizer, device)

    print(f"\nRESULT: {dataset} seed {seed}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"PPL: {ppl:.6f}")

    # Save result.
    output_dir = f"out/opt13b_{dataset}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "log_wanda.txt"),
        "w"
    ) as f:

        print(
            "method\tactual_sparsity\tppl_test",
            file=f
        )

        print(
            f"wanda\t{sparsity:.4f}\t{ppl:.4f}",
            file=f
        )

    torch.cuda.empty_cache()

    return ppl


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-1.3b",
        help="OPT model"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed"
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to run"
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )

    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Sparsity level"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="llm_weights",
        help="Model cache directory"
    )

    args = parser.parse_args()

    print("Loading LLM...")
    print(f"Model: {args.model}")
    print(f"Seeds: {args.seed} to {args.seed + args.num_seeds - 1}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Sparsity: {args.sparsity_ratio}")

    model = get_llm(
        args.model,
        args.cache_dir
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False
    )

    device = torch.device("cuda:0")

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    print("Using device:", device)

    print("\nSaving original model weights to CPU...")
    original_weights = save_original_weights(model)
    print("Original weights saved.")

    results = {
        "c4": {},
        "wikitext2": {}
    }

    for seed in range(
        args.seed,
        args.seed + args.num_seeds
    ):

        for dataset in ["c4", "wikitext2"]:

            ppl = run_experiment(
                model,
                tokenizer,
                device,
                original_weights,
                args,
                dataset,
                seed
            )

            results[dataset][seed] = ppl

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    for dataset in ["c4", "wikitext2"]:

        print(f"\n{dataset.upper()}")

        for seed in results[dataset]:

            print(
                f"Seed {seed:2d}: "
                f"{results[dataset][seed]:.6f}"
            )


if __name__ == "__main__":
    main()