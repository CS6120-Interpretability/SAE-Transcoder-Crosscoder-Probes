# Legacy, model-specific SAE helpers preserved for paper reproduction.

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from handle_sae_bench_saes import get_gemma_2_2b_sae_ids, load_gemma_2_2b_sae
from utils_data import get_xy_OOD, get_xy_glue, get_xyvals


def get_gemma_2_9b_sae_ids(layer: int):
    """
    List pretrained Gemma-2-9B SAE identifiers for a given layer, preferring the
    widest SAE with the largest L0 at non-standard layers.
    """
    all_gemma_scope_saes = get_pretrained_saes_directory()["gemma-scope-9b-pt-res"].saes_map
    all_sae_ids = [sae_id for sae_id in all_gemma_scope_saes if sae_id.split("/")[0] == f"layer_{layer}"]

    # If layer is not 20, we only keep the width_16k sae with the largest l0
    if layer != 20:
        all_sae_ids = [sae_id for sae_id in all_sae_ids if "width_16k" in sae_id]
        l0s = [int(sae_id.split("/")[2].split("_")[-1]) for sae_id in all_sae_ids]
        max_l0_index = np.argmax(l0s)
        all_sae_ids = [all_sae_ids[max_l0_index]]
        print(f"Only using the width_16k sae with the largest l0 for non-standard layer {layer}: {all_sae_ids}")

    return all_sae_ids


def get_gemma_2_9b_sae_ids_largest_l0s(layer: int, width_restriction=("16k", "131k", "1m")):
    """Return Gemma-2-9B SAE ids filtered to the widest L0 per width bucket."""
    all_sae_ids = get_gemma_2_9b_sae_ids(layer)
    width_to_largest_sae_id = {}
    width_to_largest_l0 = {}
    for sae_id in all_sae_ids:
        width = sae_id.split("/")[1].split("_")[-1]
        l0 = sae_id.split("/")[2].split("_")[-1]
        if width not in width_restriction:
            continue
        if width not in width_to_largest_sae_id:
            width_to_largest_sae_id[width] = sae_id
            width_to_largest_l0[width] = l0
        elif int(l0) > int(width_to_largest_l0[width]):
            width_to_largest_sae_id[width] = sae_id
            width_to_largest_l0[width] = l0
    return list(width_to_largest_sae_id.values())


def load_gemma_2_9b_sae(sae_id: str) -> SAE:
    """Load a Gemma-2-9B SAE onto CPU."""
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res",
        sae_id=sae_id,
        device="cpu",
    )
    return sae


def load_llama_3_1_8b_sae(sae_id: str) -> SAE:
    """Load a Llama 3.1 8B SAE onto CPU."""
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="llama_scope_lxr_32x",
        sae_id=sae_id,
        device="cpu",
    )
    return sae


def layer_to_sae_ids(layer: int, model_name: str):
    """Map model + layer to the relevant SAE identifiers."""
    if model_name == "gemma-2-9b":
        return get_gemma_2_9b_sae_ids(layer)
    if model_name == "llama-3.1-8b":
        return [f"l{layer}r_32x"]
    if model_name == "gemma-2-2b":
        assert layer == 12
        return get_gemma_2_2b_sae_ids(layer)
    raise ValueError(f"Invalid model name: {model_name}")


def sae_id_to_sae(sae_id, model_name: str, device: str):
    """Load the SAE corresponding to ``sae_id`` for the given model."""
    if model_name == "gemma-2-9b":
        return load_gemma_2_9b_sae(sae_id).to(device)
    if model_name == "llama-3.1-8b":
        return load_llama_3_1_8b_sae(sae_id).to(device)
    if model_name == "gemma-2-2b":
        return load_gemma_2_2b_sae(sae_id).to(device)
    raise ValueError(f"Invalid model name: {model_name}")


def get_xy_OOD_sae(
    dataset: str,
    k: int = 128,
    model_name: str = "gemma-2-9b",
    layer: int = 20,
    return_indices: bool = False,
    num_train: int = 1024,
):
    """
    Load SAE activations for in-distribution and OOD splits, select the top-k
    latents by mean difference, and optionally return the chosen latent indices.
    """
    _, y_test = get_xy_OOD(dataset)
    _, y_train = get_xyvals(dataset, layer=layer, model_name=model_name, MAX_AMT=1500)
    X_test = torch.load(
        f'data/sae_activations_{model_name}_OOD/{dataset}_OOD.pt',
        weights_only=False,
    ).to_dense().cpu()
    X_train = torch.load(
        f'data/sae_activations_{model_name}/{dataset}.pt',
        weights_only=True,
    ).to_dense().cpu()
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    pos_selected = pos_indices[:num_train // 2]
    neg_selected = neg_indices[:num_train // 2]

    selected_indices = np.concatenate([pos_selected, neg_selected])
    shuffled_indices = np.random.permutation(selected_indices)

    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    top_by_average_diff = sorted_indices[:k]
    X_train_filtered = X_train[:, top_by_average_diff]
    X_test_filtered = X_test[:, top_by_average_diff]
    if return_indices:
        return X_train_filtered, y_train, X_test_filtered, y_test, top_by_average_diff
    return X_train_filtered, y_train, X_test_filtered, y_test


def get_xy_glue_sae(toget: str = "ensemble", k: int = 128):
    """
    Load SAE activations for GLUE-CoLA and select the strongest latents via
    class-mean difference. When ``k==1`` the function pinches a known grammar
    feature.
    """
    dataset = "87_glue_cola"
    _, y_test = get_xy_glue(toget=toget)
    _, y_train = get_xyvals(dataset, layer=20, model_name="gemma-2-9b", MAX_AMT=1500)
    X_test = torch.load("data/dataset_investigate/sae_gemma-2-9b_87_glue_cola.pt", weights_only=False).to_dense().cpu()
    X_train = torch.load(f"data/sae_activations_gemma-2-9b_1m/{dataset}.pt", weights_only=True).to_dense().cpu()
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    pos_selected = pos_indices[:512]
    neg_selected = neg_indices[:512]

    selected_indices = np.concatenate([pos_selected, neg_selected])
    shuffled_indices = np.random.permutation(selected_indices)

    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    if k == 1:
        top_by_average_diff = sorted_indices[2:3]
        print(top_by_average_diff)
    else:
        top_by_average_diff = sorted_indices[:k]
    X_train_filtered = X_train[:, top_by_average_diff]
    X_test_filtered = X_test[:, top_by_average_diff]
    return X_train_filtered, y_train, X_test_filtered, y_test


def get_sae_layers_extra(model_name: str):
    assert model_name == "gemma-2-9b"
    return [9, 20, 31, 41]


def get_sae_layers(model_name: str):
    if model_name == "gemma-2-9b":
        return [20]
    if model_name == "llama-3.1-8b":
        return [16]
    if model_name == "gemma-2-2b":
        return [12]
    raise ValueError(f"Unknown model name: {model_name}. Pass explicit layers when using custom models.")
