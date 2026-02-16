# %%

# ------------------------------------------------------------------------------------------------
# PART 1: Generate model activations
# ------------------------------------------------------------------------------------------------

import glob
import os
import random
import argparse
import importlib
from typing import List, Optional

import pandas as pd
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm
from sae_lens import SAE

from utils_data import get_dataset_sizes, get_numbered_binary_tags, get_yvals, get_train_test_indices, resolve_model_id
from utils_sae import build_sae_description, get_sae_hook_name, infer_layer_label
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
import einops

torch.set_grad_enabled(False)

data_dir = "/mnt/sdb/jengels/data"

model_name = "gemma-2-9b"
model_id = None
device = "cuda:1"
max_seq_len = 256
layer = 20
resid_hook = "hook_resid_post"
hook_name = None
sae_release = "gemma-scope-9b-pt-res"
sae_ids = ["layer_20/width_16k/average_l0_408", "layer_20/width_16k/average_l0_68"]

# %%

parser = argparse.ArgumentParser(description="Generate multi-token model and SAE activations.")
parser.add_argument("--data_dir", type=str, default=data_dir)
parser.add_argument("--model_name", type=str, default=model_name, help="Name used for paths and default aliases.")
parser.add_argument("--model_id", type=str, default=None, help="TransformerLens/HF model id override.")
parser.add_argument("--device", type=str, default=device)
parser.add_argument("--max_seq_len", type=int, default=max_seq_len)
parser.add_argument("--layer", type=int, default=layer)
parser.add_argument("--resid_hook", type=str, default=resid_hook)
parser.add_argument("--hook_name", type=str, default=None, help="Explicit hook name for activations.")
parser.add_argument("--sae_release", type=str, default=sae_release)
parser.add_argument("--sae_ids", type=str, nargs="+", default=sae_ids)
parser.add_argument("--sae_converter", type=str, default=None, help="Converter function in module:function form.")
args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
model_id = args.model_id
device = args.device
max_seq_len = args.max_seq_len
layer = args.layer
resid_hook = args.resid_hook
hook_name = args.hook_name
sae_release = args.sae_release
sae_ids = args.sae_ids
sae_converter = args.sae_converter

os.makedirs(f"{data_dir}/model_activations_{model_name}_{max_seq_len}", exist_ok=True)

model_id = resolve_model_id(model_name, model_id_override=model_id)
model = HookedTransformer.from_pretrained(model_id, device=device)

# %%

# Important to ensure correct token is at the correct position, either at the text_length position or at the end of the sequence
tokenizer = model.tokenizer
tokenizer.truncation_side='left'
tokenizer.padding_side='right'

if hook_name is None:
    hook_name = f"blocks.{layer}.{resid_hook}"

# %%

dataset_names = glob.glob("data/cleaned_data/*.csv")

# Randomize dataset names so multiple GPUs can work on it
random.shuffle(dataset_names)

for dataset_name in dataset_names:
    file_path = f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset_name.split('/')[-1].split('.')[0]}_{hook_name}.pt"
    if os.path.exists(file_path):
        print(f"Skipping {dataset_name} because activations already exist")
        continue
    dataset = pd.read_csv(dataset_name)
    dataset_short_name = dataset_name.split("/")[-1].split(".")[0]

    text = dataset["prompt"].tolist()
    
    text_lengths = []
    for t in text:
        text_lengths.append(len(tokenizer(t)['input_ids']))

    print(f"Generating activations for {dataset_short_name} (no existing activations)")

    batch_size = 1
    all_activations = []
    bar = tqdm(range(0, len(text), batch_size))
    for i in bar:
        batch_text = text[i:i+batch_size]
        batch_lengths = text_lengths[i:i+batch_size]
        batch = tokenizer(batch_text, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors="pt",)
        batch = batch.to(device)
        logits, cache = model.run_with_cache(batch["input_ids"], names_filter=hook_name)
        for j, length in enumerate(batch_lengths):
            activations = cache[hook_name][:, :].cpu()[0]
            actual_length = min(length, max_seq_len)
            activations[actual_length:] = 0
            all_activations.append(activations)
        bar.set_description(f"{len(all_activations)}")

    torch.save(torch.stack(all_activations), file_path)
# %%

# ------------------------------------------------------------------------------------------------
# PART 2: Generate SAE activations
# ------------------------------------------------------------------------------------------------

# %%

dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()

# %%

def get_sae_paths(dataset, layer_label, sae_id):
    os.makedirs(f"{data_dir}/sae_probes_{model_name}_{max_seq_len}", exist_ok=True)
    os.makedirs(f"{data_dir}/sae_activations_{model_name}_{max_seq_len}", exist_ok=True)
    description_string = build_sae_description(dataset, sae_id, layer_label)
    train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{description_string}_X_train_sae.pt"
    test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{description_string}_X_test_sae.pt"
    y_train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{description_string}_y_train.pt"
    y_test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{description_string}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def save_activations(path, activation):
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)

def save_with_sae(layer_label, sae, hook_name):
    for dataset in datasets:
        paths = get_sae_paths(dataset, layer_label, sae_id)
        train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
        
        if os.path.exists(train_path):
            continue
        
        size = dataset_sizes[dataset]
        num_train = min(size-100, 1024)
        num_test = size - num_train
        y = get_yvals(dataset)
        train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)

        try:
            X_tensor = torch.load(
                f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_{hook_name}.pt",
                weights_only=True,
            )
            print(X_tensor.shape)
        except Exception as e:
            print(f"Error loading {dataset}.{hook_name}.pt: {e}")
            continue

        # X_tensor is shape (num_samples, seq_len, hidden_size)

        x_shape = X_tensor.shape
        flattened_x = X_tensor.flatten(end_dim=1)

        all_x_sae = []
        batch_size = 1024

        for i in tqdm(range(0, len(flattened_x), batch_size)):
            batch = flattened_x[i:i+batch_size].to(device)
            all_x_sae.append(sae.encode(batch).cpu())
        all_x_sae = torch.cat(all_x_sae)

        # all_x_sae is shape (num_samples, sae_hidden_size)

        all_x_sae = einops.rearrange(all_x_sae, "(b s) d -> b s d", b=x_shape[0], s=x_shape[1])

        print(all_x_sae.shape)

        # Split into train and test
        X_train_sae = all_x_sae[train_indices]
        X_test_sae = all_x_sae[test_indices]

        print(X_train_sae.shape)
        print(X_test_sae.shape)

        y_train = y[train_indices]
        y_test = y[test_indices]

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))

# Only run on one SAE, for memory reasons we add the loop outside the process
layer_label = infer_layer_label(layer, hook_name)
for sae_id in sae_ids:
    converter_fn = None
    if sae_converter:
        module_path, func_name = sae_converter.split(":", 1)
        converter_fn = getattr(importlib.import_module(module_path), func_name)
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
        converter=converter_fn,
    )[0]
    try:
        sae_hook = get_sae_hook_name(sae)
    except ValueError:
        sae_hook = hook_name
    if sae_hook != hook_name:
        print(f"Warning: SAE hook_name {sae_hook} != requested hook {hook_name}.")
    print(f"Generating SAE data for layer {layer_label}, SAE {sae_id}")
    save_with_sae(layer_label, sae, hook_name)

# %%
