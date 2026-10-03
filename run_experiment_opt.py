import argparse
import csv
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
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(original_weights[name].to(param.device))


def save_result(csv_path, model_name, sparsity, dataset, seed, ppl):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model",
                "sparsity",
                "dataset",
                "seed",
                "ppl"
            ])

        writer.writerow([
            model_name,
            sparsity,
            dataset,
            seed,
            ppl
        ])


def run_experiment(
    model,
    tokenizer,
    device,
    original_weights,
    args,
    dataset,
    seed,
    csv_path
):

    print("\n" + "=" * 70)
    print(f"STARTING | {dataset} | SEED {seed}")
    print("=" * 70)

    # Restore the original unpruned model.
    restore_original_weights(model, original_weights)

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.calib_dataset = dataset
    args.seed = seed

    prune_wanda(
        args,
        model,
        tokenizer,
        device,
        prune_n=0,
        prune_m=0
    )

    sparsity = check_sparsity(model)
    ppl = eval_ppl(args, model, tokenizer, device)

    print(f"\nRESULT: {dataset} seed {seed}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"PPL: {ppl:.6f}")

    # Save individual result.
    output_dir = os.path.join(
        "out",
        f"{args.model.split('/')[-1]}_s{args.sparsity_ratio:.1f}",
        f"{dataset}_seed{seed}"
    )

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

    # IMPORTANT:
    # Write to the aggregate CSV immediately.
    # Therefore, completed experiments are preserved even if
    # the program crashes during a later experiment.
    save_result(
        csv_path,
        args.model,
        args.sparsity_ratio,
        dataset,
        seed,
        ppl
    )

    print(f"Saved result to: {csv_path}")

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

    # Wanda settings used by prune_wanda.
    args.sparsity_type = "unstructured"
    args.prune_method = "wanda"
    args.use_variant = False

    first_seed = args.seed
    last_seed = args.seed + args.num_seeds - 1

    model_short_name = args.model.split("/")[-1]

    # Create a unique CSV filename based on:
    # model + sparsity + seed range
    csv_dir = "results"
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = (
        f"{model_short_name}"
        f"_s{args.sparsity_ratio:.1f}"
        f"_seed{first_seed}-{last_seed}.csv"
    )

    csv_path = os.path.join(
        csv_dir,
        csv_filename
    )

    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Model:              {args.model}")
    print(f"Seeds:              {first_seed} to {last_seed}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Sparsity:           {args.sparsity_ratio}")
    print(f"Results CSV:        {csv_path}")
    print("=" * 70)

    print("\nLoading LLM...")
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

    # Run every seed for both calibration datasets.
    for seed in range(first_seed, last_seed + 1):

        for dataset in ["c4", "wikitext2"]:

            run_experiment(
                model=model,
                tokenizer=tokenizer,
                device=device,
                original_weights=original_weights,
                args=args,
                dataset=dataset,
                seed=seed,
                csv_path=csv_path
            )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()