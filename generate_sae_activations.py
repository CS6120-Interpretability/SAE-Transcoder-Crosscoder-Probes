# %%
import argparse
import importlib
import os
import random
import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning
from utils_data import (
    get_OOD_datasets,
    get_dataset_sizes, 
    get_numbered_binary_tags, 
    get_xy_traintest, 
    get_xy_traintest_specify,
    get_training_sizes,
    get_class_imbalance,
    get_classimabalance_num_train,
)
from utils_sae import build_sae_description, get_sae_hook_name, infer_layer_label, parse_layers_from_sae_id
from legacy_sae import get_sae_layers, get_sae_layers_extra, layer_to_sae_ids, sae_id_to_sae

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# %%

# Common variables
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()

# %%
# Helper functions for all settings
def save_activations(path: str, activation: torch.Tensor) -> None:
    """Save activations in sparse format to save space."""
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)

def load_activations(path: str) -> torch.Tensor:
    """Load activations from sparse format."""
    return torch.load(path, weights_only=True).to_dense().float()

# %%
# Normal setting functions
def get_sae_paths_normal(dataset: str, layer: Union[int, str], sae_id: str, model_name: str = "gemma-2-9b") -> Dict[str, str]:
    """Get paths for normal setting"""
    os.makedirs(f"data/sae_probes_{model_name}/normal_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/normal_setting", exist_ok=True)

    description_string = build_sae_description(dataset, sae_id, layer)

    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def save_with_sae_normal(
    layer: Union[int, str],
    sae: torch.nn.Module,
    sae_id: str,
    model_name: str,
    device: str,
    hook_names: Optional[Sequence[str]] = None,
) -> None:
    """Generate and save SAE activations for normal setting"""
    for dataset in datasets:
        paths = get_sae_paths_normal(dataset, layer, sae_id, model_name)
        train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
        
        all_paths_exist = all([os.path.exists(train_path), os.path.exists(test_path), os.path.exists(y_train_path), os.path.exists(y_test_path)])
        if all_paths_exist:
            continue
        
        size = dataset_sizes[dataset]
        num_train = min(size-100, 1024)
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train,
            dataset,
            layer,
            model_name=model_name,
            hook_name=hook_names,
        )

        batch_size = 128
        X_train_sae = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            X_train_sae.append(sae.encode(batch).cpu())
        X_train_sae = torch.cat(X_train_sae)

        X_test_sae = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            X_test_sae.append(sae.encode(batch).cpu())
        X_test_sae = torch.cat(X_test_sae)

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))

# %%
# Data scarcity setting functions
def get_sae_paths_scarcity(dataset: str, layer: Union[int, str], sae_id: str, num_train: int, model_name: str = "gemma-2-9b") -> Dict[str, str]:
    """Get paths for data scarcity setting"""
    os.makedirs(f"data/sae_probes_{model_name}/scarcity_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/scarcity_setting", exist_ok=True)
    
    description_string = build_sae_description(dataset, sae_id, layer)

    train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_test.pt"
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def save_with_sae_scarcity(
    layer: Union[int, str],
    sae: torch.nn.Module,
    sae_id: str,
    model_name: str,
    device: str,
    hook_names: Optional[Sequence[str]] = None,
) -> None:
    """Generate and save SAE activations for data scarcity setting"""
    train_sizes = get_training_sizes()
    
    for dataset in datasets:
        for num_train in train_sizes:
            if num_train > dataset_sizes[dataset] - 100:
                continue
                
            paths = get_sae_paths_scarcity(dataset, layer, sae_id, num_train, model_name)
            train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
            
            if all(os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]):
                continue

            X_train, y_train, X_test, y_test = get_xy_traintest(
                num_train,
                dataset,
                layer,
                model_name=model_name,
                hook_name=hook_names,
            )

            batch_size = 128
            X_train_sae = []
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i:i+batch_size].to(device)
                X_train_sae.append(sae.encode(batch).cpu())
            X_train_sae = torch.cat(X_train_sae)

            X_test_sae = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size].to(device)
                X_test_sae.append(sae.encode(batch).cpu())
            X_test_sae = torch.cat(X_test_sae)

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))

# %%
# Class imbalance setting functions
def get_sae_paths_imbalance(dataset: str, layer: Union[int, str], sae_id: str, frac: float, model_name: str = "gemma-2-9b") -> Dict[str, str]:
    """Get paths for class imbalance setting"""
    os.makedirs(f"data/sae_probes_{model_name}/class_imbalance_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/class_imbalance_setting", exist_ok=True)
    
    description_string = build_sae_description(dataset, sae_id, layer)
        
    train_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def save_with_sae_imbalance(
    layer: Union[int, str],
    sae: torch.nn.Module,
    sae_id: str,
    model_name: str,
    device: str,
    hook_names: Optional[Sequence[str]] = None,
) -> None:
    """Generate and save SAE activations for class imbalance setting"""
    fracs = get_class_imbalance()
    
    for dataset in datasets:
        for frac in fracs:
            paths = get_sae_paths_imbalance(dataset, layer, sae_id, frac, model_name)
            train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
            
            if os.path.exists(train_path):
                continue
            
            num_train, num_test = get_classimabalance_num_train(dataset)
            X_train, y_train, X_test, y_test = get_xy_traintest_specify(
                num_train,
                dataset,
                layer,
                pos_ratio=frac,
                model_name=model_name,
                num_test=num_test,
                hook_name=hook_names,
            )

            batch_size = 128
            X_train_sae = []
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i:i+batch_size].to(device)
                X_train_sae.append(sae.encode(batch).cpu())
            X_train_sae = torch.cat(X_train_sae)

            X_test_sae = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size].to(device)
                X_test_sae.append(sae.encode(batch).cpu())
            X_test_sae = torch.cat(X_test_sae)

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))


def save_with_sae_ood(
    layer: Union[int, str],
    sae: torch.nn.Module,
    sae_id: str,
    model_name: str,
    device: str,
    hook_names: Optional[Sequence[str]] = None,
) -> None:
    """Generate and save SAE activations for OOD setting"""
    for dataset in get_OOD_datasets():
        paths = get_sae_paths_ood(dataset, layer, sae_id, model_name)
        train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
        
        if all(os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]):
            continue
        
        X_train, y_train, X_test, y_test = get_xy_traintest(
            1024,
            dataset,
            layer,
            model_name=model_name,
            hook_name=hook_names,
        )
        
        batch_size = 128
        X_train_sae = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            X_train_sae.append(sae.encode(batch).cpu())
        X_train_sae = torch.cat(X_train_sae)

        X_test_sae = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            X_test_sae.append(sae.encode(batch).cpu())
        X_test_sae = torch.cat(X_test_sae)

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))
        
        
        
        

# %%
# OOD setting functions
def get_sae_paths_ood(dataset: str, layer: Union[int, str], sae_id: str, model_name: str = "gemma-2-9b") -> Dict[str, str]:
    os.makedirs(f"data/sae_probes_{model_name}/ood_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/ood_setting", exist_ok=True)

    description_string = build_sae_description(dataset, sae_id, layer)
    
    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/OOD_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/OOD_setting/{description_string}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

# %%
# Process SAEs for a specific model and setting
def process_model_setting(
    model_name: str,
    setting: str,
    device: str,
    randomize_order: bool,
    sae_release: Optional[str] = None,
    sae_ids_override: Optional[List[str]] = None,
    hook_name_override: Optional[str] = None,
    hook_names_override: Optional[List[str]] = None,
    resid_hook: str = "hook_resid_post",
    layers_override: Optional[List[int]] = None,
    sae_converter: Optional[str] = None,
) -> bool:
    """
    Generate SAE activations for a model/setting combination, skipping SAEs that
    already have cached outputs. Returns True when more work remains so callers
    can loop until all SAEs are processed.

    Args:
        model_name: Model name used for activation paths.
        setting: Experiment setting (normal, scarcity, imbalance, OOD).
        device: Device to run SAE encoding on.
        randomize_order: Shuffle SAE list to aid multi-process execution.
        sae_release: Optional SAE Lens release name for explicit SAE ids.
        sae_ids_override: Optional explicit SAE ids to process.
        hook_name_override: Optional explicit hook name to use for activations.
        hook_names_override: Optional explicit hook names for multi-layer SAEs.
        resid_hook: Residual hook suffix for layer-derived hook names.
        layers_override: Optional layer indices to override default lists.
        sae_converter: Optional converter function in module:function form.
    """
    print(f"Running SAE activation generation for {model_name} in {setting} setting")
    
    supported_models = {"gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"}
    if sae_ids_override is not None and sae_release is None and model_name not in supported_models:
        raise ValueError("Custom SAE ids require --sae_release when model_name is not one of the built-ins.")

    if layers_override is not None:
        layers = layers_override
    elif model_name in supported_models:
        layers = get_sae_layers(model_name)
    else:
        layers = [None]
    if model_name == "gemma-2-9b" and setting == "normal" and layers_override is None:
        layers = get_sae_layers_extra(model_name)
    found_missing = False
    
    for layer in layers:
        if sae_ids_override is not None:
            sae_ids = sae_ids_override
        else:
            sae_ids = layer_to_sae_ids(layer, model_name)
            if model_name == "gemma-2-9b" and setting != "normal":
                sae_ids = ["layer_20/width_16k/average_l0_408", "layer_20/width_131k/average_l0_276", "layer_20/width_1m/average_l0_193"]
            
        if randomize_order:
            random.shuffle(sae_ids)
        
        for sae_id in sae_ids:
            print(f"Processing SAE: {sae_id}")
            
            # Check if we need to generate activations for this SAE
            missing_data = False
            try:
                if sae_release is None:
                    sae = sae_id_to_sae(sae_id, model_name, device)
                else:
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
            except Exception as e:
                print(f"Error loading SAE {sae_id}: {e}")
                continue

            hook_names: Optional[List[str]] = None
            if hook_names_override:
                hook_names = hook_names_override
            elif hook_name_override:
                hook_names = [hook_name_override]
            else:
                try:
                    hook_names = [get_sae_hook_name(sae)]
                except ValueError:
                    hook_names = None

            if hook_names is None:
                layers_from_id = parse_layers_from_sae_id(str(sae_id))
                if layers_from_id:
                    hook_names = [f"blocks.{layer_idx}.{resid_hook}" for layer_idx in layers_from_id]

            if hook_names is None:
                raise ValueError("Unable to infer hook_name(s); pass --hook_name/--hook_names or --layers for this SAE.")

            layer_label = infer_layer_label(layer, hook_names)
            
            if setting == "normal":
                # Check normal setting
                for dataset in datasets:
                    paths = get_sae_paths_normal(dataset, layer_label, sae_id, model_name)
                    if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                          paths["y_train_path"], paths["y_test_path"]]):
                        print(f"Missing data for dataset {dataset}")
                        missing_data = True
                        break
                
                if missing_data:
                    try:
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_normal(layer_label, sae, sae_id, model_name, device, hook_names=hook_names)
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue
            
            elif setting == "scarcity":
                # Check data scarcity setting
                train_sizes = get_training_sizes()
                for dataset in datasets:
                    for num_train in train_sizes:
                        if num_train > dataset_sizes[dataset] - 100:
                            continue
                        paths = get_sae_paths_scarcity(dataset, layer_label, sae_id, num_train, model_name)
                        if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                              paths["y_train_path"], paths["y_test_path"]]):
                            print(f"Missing data for dataset {dataset}, num_train {num_train}")
                            missing_data = True
                            break
                    if missing_data:
                        break
                
                if missing_data:
                    try:
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_scarcity(layer_label, sae, sae_id, model_name, device, hook_names=hook_names)
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue
            
            elif setting == "imbalance":
                # Check class imbalance setting
                fracs = get_class_imbalance()
                for dataset in datasets:
                    for frac in fracs:
                        paths = get_sae_paths_imbalance(dataset, layer_label, sae_id, frac, model_name)
                        if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                              paths["y_train_path"], paths["y_test_path"]]):
                            print(f"Missing data for dataset {dataset}, frac {frac}")
                            missing_data = True
                            break
                    if missing_data:
                        break
                
                if missing_data:
                    try:
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_imbalance(layer_label, sae, sae_id, model_name, device, hook_names=hook_names)
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue

            elif setting == "OOD":
                # Check OOD setting
                for dataset in get_OOD_datasets():
                    paths = get_sae_paths_ood(dataset, layer_label, sae_id, model_name)
                    if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                          paths["y_train_path"], paths["y_test_path"]]):
                        print(f"Missing data for dataset {dataset}")
                        missing_data = True
                        break

                    if missing_data:
                        break
                
                if missing_data:
                    try:
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_ood(layer_label, sae, sae_id, model_name, device, hook_names=hook_names)
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue
                
            else:
                raise ValueError(f"Invalid setting: {setting}")
       
        if found_missing:
            break
    
    if not found_missing:
        print(f"All SAE activations for {model_name} in {setting} setting have been generated!")
    
    return found_missing

# %%
# Main function to process all models and settings
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--setting", type=str, default=None, 
                        choices=["normal", "scarcity", "imbalance", "OOD"])
    parser.add_argument("--randomize_order", action="store_true", help="Randomize the order of datasets and settings, useful for parallelizing")
    parser.add_argument("--sae_release", type=str, default=None, help="SAE Lens release name when using explicit SAE ids.")
    parser.add_argument("--sae_ids", type=str, nargs="+", default=None, help="Explicit SAE ids to process.")
    parser.add_argument("--hook_name", type=str, default=None, help="Explicit hook name to use for activations.")
    parser.add_argument("--hook_names", type=str, nargs="+", default=None, help="Explicit hook names for multi-layer SAEs.")
    parser.add_argument("--resid_hook", type=str, default="hook_resid_post", help="Residual hook suffix for parsed layer lists.")
    parser.add_argument("--sae_converter", type=str, default=None, help="Converter function in module:function form.")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Override layer list when model defaults are not desired.")

    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    setting = args.setting
    randomize_order = args.randomize_order
    sae_release = args.sae_release
    sae_ids_override = args.sae_ids
    hook_name_override = args.hook_name
    hook_names_override = args.hook_names
    resid_hook = args.resid_hook
    sae_converter = args.sae_converter
    layers_override = args.layers

    model_names = ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]
    settings = ["normal", "scarcity", "imbalance", "OOD"]
    
    # If specific model and setting are provided via command line, use those and only run a max of one sae
    # This helps with memory and parallelization
    if model_name is not None and setting is not None:
        process_model_setting(
            model_name,
            setting,
            device,
            randomize_order,
            sae_release=sae_release,
            sae_ids_override=sae_ids_override,
            hook_name_override=hook_name_override,
            hook_names_override=hook_names_override,
            resid_hook=resid_hook,
            layers_override=layers_override,
            sae_converter=sae_converter,
        )
        exit(0)

    # Otherwise, loop through all models and settings
    for curr_model_name in model_names:
        if randomize_order:
            random.shuffle(settings)
        for curr_setting in settings:
            print(f"\n{'='*50}")
            print(f"Processing {curr_model_name} in {curr_setting} setting")
            print(f"{'='*50}\n")
            do_loop = True
            while do_loop:
                do_loop = process_model_setting(
                    curr_model_name,
                    curr_setting,
                    device,
                    randomize_order,
                    sae_release=sae_release,
                    sae_ids_override=sae_ids_override,
                    hook_name_override=hook_name_override,
                    hook_names_override=hook_names_override,
                    resid_hook=resid_hook,
                    layers_override=layers_override,
                    sae_converter=sae_converter,
                )
