import argparse
import csv
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import random

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

def load_calibration_datasets():
    print("Loading C4 dataset...")
    c4_data = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz",
        split="train"
    )

    print("Loading WikiText-2 dataset...")
    wikitext_data = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        split="train"
    )

    return c4_data, wikitext_data


def get_calibration_data(dataset, data, nsamples, seed, seqlen, tokenizer):

    random.seed(seed)

    trainloader = []

    if dataset == "c4":

        for _ in range(nsamples):

            while True:
                i = random.randint(0, len(data) - 1)

                trainenc = tokenizer(
                    data[i]["text"],
                    return_tensors="pt"
                )

                if trainenc.input_ids.shape[1] >= seqlen:
                    break

            max_start = trainenc.input_ids.shape[1] - seqlen
            start = random.randint(0, max_start)

            inp = trainenc.input_ids[:, start:start + seqlen]

            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

        return trainloader

    else:

        trainenc = tokenizer(
            " ".join(data["text"]),
            return_tensors="pt"
        )

        for _ in range(nsamples):

            i = random.randint(
                0,
                trainenc.input_ids.shape[1] - seqlen
            )

            j = i + seqlen

            inp = trainenc.input_ids[:, i:j]

            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

        return trainloader

def run_experiment(
    model,
    tokenizer,
    device,
    original_weights,
    args,
    dataset,
    seed,
    csv_path,
    calibration_datasets
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

    calib_data = get_calibration_data(
        dataset,
        calibration_datasets[dataset],
        args.nsamples,
        seed,
        model.seqlen,
        tokenizer
    )

    prune_wanda(
        args,
        model,
        tokenizer,
        device,
        prune_n=0,
        prune_m=0,
        dataloader=calib_data
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

    print("\nLoading calibration datasets...")
    c4_data, wikitext_data = load_calibration_datasets()

    calibration_datasets = {
        "c4": c4_data,
        "wikitext2": wikitext_data
    }

    print("Calibration datasets loaded.")

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
                csv_path=csv_path,
                calibration_datasets=calibration_datasets
            )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()