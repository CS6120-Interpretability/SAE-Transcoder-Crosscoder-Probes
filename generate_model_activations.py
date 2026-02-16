# %%

import glob
import os
import random
import argparse
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
from utils_data import resolve_model_id

torch.set_grad_enabled(False)


def resolve_hook_names(
    model: HookedTransformer,
    hook_names: Optional[List[str]],
    layers: Optional[List[int]],
    include_embed: bool,
    resid_hook: str,
    num_default_layers: int,
) -> List[str]:
    """
    Determine which hook names to cache, either from explicit names, explicit
    layers, or a default evenly spaced layer selection.
    """
    if hook_names:
        return hook_names

    if layers is None:
        n_layers = getattr(model.cfg, "n_layers", None)
        if n_layers is None:
            raise ValueError("Model config missing n_layers; please pass --layers or --hook_names.")
        if num_default_layers < 1:
            raise ValueError("num_default_layers must be >= 1.")
        if num_default_layers == 1:
            layers = [n_layers - 1]
        else:
            step = max(1, (n_layers - 1) // (num_default_layers - 1))
            layers = list(range(0, n_layers, step))[:num_default_layers]

    hook_list = []
    if include_embed:
        hook_list.append("hook_embed")
    hook_list.extend([f"blocks.{layer}.{resid_hook}" for layer in layers])
    return hook_list

def generate_dataset_activations(
    model_name: str,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
    OOD: bool = False,
    hook_names: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    include_embed: bool = True,
    resid_hook: str = "hook_resid_post",
    num_default_layers: int = 4,
    model_id: Optional[str] = None,
) -> None:
    """
    Run the requested Transformer through every probing dataset and persist the
    cached activations for the chosen hooks.

    Args:
        model_name: Path-friendly identifier used in output directories.
        device: Torch device string to place the model and tokenized batches on.
        max_seq_len: Maximum sequence length used when tokenizing each prompt.
        OOD: When True, read prompts from ``data/OOD data`` instead of the
            standard cleaned datasets.
        hook_names: Optional explicit hook names to cache (overrides ``layers``).
        layers: Optional residual stream layer indices to cache.
        include_embed: Whether to also cache ``hook_embed`` activations.
        resid_hook: Hook suffix to use for layer-based hooks (e.g. ``hook_resid_post``).
        num_default_layers: Number of evenly spaced layers to use when no
            ``hook_names`` or ``layers`` are provided.
        model_id: Optional TransformerLens/HF model id override.
    """
    os.makedirs(
        f"data/model_activations_{model_name}{'_OOD' if OOD else ''}",
        exist_ok=True,
    )

    # Load the model (allow arbitrary TransformerLens identifiers)
    model_id = resolve_model_id(model_name, model_id_override=model_id)
    model = HookedTransformer.from_pretrained(model_id, device=device)

    # Important to ensure correct token is at the correct position, either at the text_length position or at the end of the sequence
    tokenizer = model.tokenizer
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'right'

    # Preserve legacy defaults for the three paper models if no overrides provided.
    legacy_layers = {
        "gemma-2-9b": [9, 20, 31, 41],
        "llama-3.1-8b": [8, 16, 24, 31],
        "gemma-2-2b": [12],
    }
    if hook_names is None and layers is None and model_name in legacy_layers:
        layers = legacy_layers[model_name]
        include_embed = True

    # Resolve hook names using explicit input or model config defaults.
    hook_names = resolve_hook_names(
        model,
        hook_names=hook_names,
        layers=layers,
        include_embed=include_embed,
        resid_hook=resid_hook,
        num_default_layers=num_default_layers,
    )

    if OOD:
        dataset_names = glob.glob("data/OOD data/*.csv")
    else:
        dataset_names = glob.glob("data/cleaned_data/*.csv")

    # Randomize dataset names so multiple GPUs can work on it
    random.shuffle(dataset_names)

    for dataset_name in dataset_names:
        dataset = pd.read_csv(dataset_name)
        if "prompt" not in dataset.columns:
            continue
        dataset_short_name = dataset_name.split("/")[-1].split(".")[0]
        file_names = [
            f"data/model_activations_{model_name}{'_OOD' if OOD else ''}/{dataset_short_name}_{hook_name}.pt"
            for hook_name in hook_names
        ]
        lengths: Optional[List[int]] = None
        if all(os.path.exists(file_name) for file_name in file_names):
            lengths = [torch.load(file_name, weights_only=True).shape[0] for file_name in file_names]

        text = dataset["prompt"].tolist()
        
        text_lengths = []
        for t in text:
            text_lengths.append(len(tokenizer(t)['input_ids']))

        if lengths is not None and all(length == len(text_lengths) for length in lengths):
            print(f"Skipping {dataset_short_name} because correct length activations already exist")
            continue

        if lengths is not None:
            print(f"Generating activations for {dataset_short_name} (bad existing activations)")
            print(lengths, len(text_lengths))
        else:
            print(f"Generating activations for {dataset_short_name} (no existing activations)")


        batch_size = 1
        all_activations: Dict[str, List[torch.Tensor]] = {hook_name: [] for hook_name in hook_names}
        bar = tqdm(range(0, len(text), batch_size))
        for i in bar:
            batch_text = text[i:i+batch_size]
            batch_lengths = text_lengths[i:i+batch_size]
            batch = tokenizer(batch_text, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
            batch = batch.to(device)
            logits, cache = model.run_with_cache(batch["input_ids"], names_filter=hook_names)
            for j, length in enumerate(batch_lengths):
                for hook_name in hook_names:
                    activation_pos = min(length - 1, max_seq_len - 1)
                    all_activations[hook_name].append(cache[hook_name][:, activation_pos].cpu())
            bar.set_description(f"{len(all_activations[hook_name])}")

        print(i, len(all_activations[hook_name]), len(torch.cat(all_activations[hook_name])))

        for hook_name, file_name in zip(hook_names, file_names):
            all_activations[hook_name] = torch.cat(all_activations[hook_name])
            torch.save(all_activations[hook_name], file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--OOD", action="store_true")
    parser.add_argument("--hook_names", type=str, nargs="+", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--no_embed", action="store_true", help="Disable the embedding hook.")
    parser.add_argument("--resid_hook", type=str, default="hook_resid_post")
    parser.add_argument("--num_default_layers", type=int, default=4)
    args = parser.parse_args()

    if args.model_name:
        # Run for a single model
        generate_dataset_activations(
            args.model_name,
            args.device,
            args.max_seq_len,
            args.OOD,
            hook_names=args.hook_names,
            layers=args.layers,
            include_embed=not args.no_embed,
            resid_hook=args.resid_hook,
            num_default_layers=args.num_default_layers,
            model_id=args.model_id,
        )
    else:
        # Run for all models
        model_names = ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]
        for model_name in model_names:
            print(f"Processing model: {model_name}")
            generate_dataset_activations(
                model_name,
                args.device,
                args.max_seq_len,
                args.OOD,
                hook_names=args.hook_names,
                layers=args.layers,
                include_embed=not args.no_embed,
                resid_hook=args.resid_hook,
                num_default_layers=args.num_default_layers,
                model_id=args.model_id,
            )
# %%
