"""
Custom loader for Google Gemma Scope 2 Crosscoder SAEs.

This loader handles the specific format used by Google's Gemma Scope 2
crosscoder SAEs on HuggingFace Hub.

Usage:
    from sae_lens import SAE
    from gemma_scope_2_crosscoder_loader import gemma_scope_2_crosscoder_huggingface_loader

    sae = SAE.from_pretrained(
        release="google/gemma-scope-2-1b-it",
        sae_id="crosscoder/layer_7_13_17_22_width_262k_l0_medium",
        device="cuda",
        converter=gemma_scope_2_crosscoder_huggingface_loader,
    )
"""

import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, list_repo_tree
from safetensors import safe_open


def load_safetensors_weights(
    path: str | Path, device: str = "cpu", dtype: str | None = None
) -> dict[str, torch.Tensor]:
    """Load safetensors weights and optionally convert to a different dtype."""
    loaded_weights = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            weight = f.get_tensor(k)
            if dtype is not None:
                # Handle dtype conversion if needed
                weight = weight.to(dtype=_str_to_dtype(dtype))
            loaded_weights[k] = weight
    return loaded_weights


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_gemma_scope_2_crosscoder_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load configuration for a Gemma Scope 2 Crosscoder SAE from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "google/gemma-scope-2-1b-it")
        folder_name: Path to the SAE folder (e.g., "crosscoder/layer_7_13_17_22_width_262k_l0_medium")
        device: Device to load on
        force_download: Force re-download
        cfg_overrides: Configuration overrides

    Returns:
        Configuration dictionary for the SAE
    """

    # Try to load config.json from the folder
    cfg_path = None
    try:
        cfg_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{folder_name}/config.json",
            force_download=force_download,
        )
    except Exception as e:
        print(
            f"Warning: Could not load config.json from {folder_name}/config.json: {e}"
        )

    # Load config from file
    if cfg_path:
        with open(cfg_path, "r") as f:
            raw_cfg_dict = json.load(f)
    else:
        # Fallback: infer configuration from folder name
        raw_cfg_dict = _infer_gemma_scope_2_crosscoder_cfg(repo_id, folder_name)

    # Parse layer information from folder name
    # Example: "crosscoder/layer_7_13_17_22_width_262k_l0_medium"
    match = re.search(r"layer_(\d+(?:_\d+)*)", folder_name)
    layers = [int(l) for l in match.group(1).split("_")] if match else []

    # Determine model name from repo_id
    if "1b-it" in repo_id:
        model_name = "google/gemma-3-1b-it"
    elif "1b-pt" in repo_id:
        model_name = "google/gemma-3-1b-pt"
    elif "4b-it" in repo_id:
        model_name = "google/gemma-3-4b-it"
    elif "4b-pt" in repo_id:
        model_name = "google/gemma-3-4b-pt"
    elif "12b-it" in repo_id:
        model_name = "google/gemma-3-12b-it"
    elif "12b-pt" in repo_id:
        model_name = "google/gemma-3-12b-pt"
    elif "27b-it" in repo_id:
        model_name = "google/gemma-3-27b-it"
    elif "27b-pt" in repo_id:
        model_name = "google/gemma-3-27b-pt"
    else:
        model_name = "google/gemma-2-2b"  # Fallback

    # Build configuration
    cfg = {
        "architecture": "jump_relu",
        "model_name": model_name,
        "device": device or "cpu",
        "dtype": "float32",
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "none",
    }

    # Add information from config file if available
    if raw_cfg_dict:
        # Extract relevant fields
        if "d_sae" in raw_cfg_dict:
            cfg["d_sae"] = raw_cfg_dict["d_sae"]
        if "d_in" in raw_cfg_dict:
            cfg["d_in"] = raw_cfg_dict["d_in"]
        if "d_out" in raw_cfg_dict:
            cfg["d_out"] = raw_cfg_dict["d_out"]
        if "dtype" in raw_cfg_dict:
            cfg["dtype"] = raw_cfg_dict["dtype"]

    # Apply any user overrides
    if cfg_overrides is not None:
        cfg.update(cfg_overrides)

    return cfg


def _infer_gemma_scope_2_crosscoder_cfg(
    repo_id: str, folder_name: str
) -> dict[str, Any]:
    """
    Infer configuration from folder name when config.json is not available.

    Crosscoder SAEs typically have dimensions based on their model size.
    """

    # Determine model dimensions based on repo_id
    if "1b" in repo_id:
        d_in = 2048  # Gemma 2 1B hidden dimension
    elif "4b" in repo_id:
        d_in = 4096  # Gemma 2 4B hidden dimension
    elif "12b" in repo_id:
        d_in = 9216  # Gemma 2 12B hidden dimension
    elif "27b" in repo_id:
        d_in = 9216  # Gemma 2 27B hidden dimension
    else:
        d_in = 2048  # Default

    # Extract width from folder name
    # Example: "crosscoder/layer_7_13_17_22_width_262k_l0_medium"
    match = re.search(r"width_(\d+)k", folder_name)
    if match:
        d_sae = int(match.group(1)) * 1000
    else:
        d_sae = 262 * 1000  # Default

    return {
        "d_in": d_in,
        "d_sae": d_sae,
        "d_out": d_in,  # Crosscoders typically have same output dimension
    }


def gemma_scope_2_crosscoder_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    """
    Load a Gemma Scope 2 Crosscoder SAE from HuggingFace.

    Handles split weight files (params_layer_0.safetensors, params_layer_1.safetensors, etc.)

    Args:
        repo_id: HuggingFace repository ID
        folder_name: Path to the SAE folder within the repo
        device: Device to load on
        force_download: Force re-download
        cfg_overrides: Configuration overrides

    Returns:
        Tuple of (config_dict, state_dict, None)
    """

    # Get configuration
    cfg_dict = get_gemma_scope_2_crosscoder_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Find all params files for this crosscoder
    print(f"Looking for weight files in {folder_name}...")
    params_files = []

    try:
        # List all files in the folder
        repo_tree = list_repo_tree(repo_id, recursive=True)
        for item in repo_tree:
            if folder_name in item.path and item.path.startswith(folder_name):
                if item.path.endswith(".safetensors") and "params" in item.path:
                    params_files.append(item.path)

        params_files.sort()
        print(f"Found {len(params_files)} weight files: {params_files}")
    except Exception as e:
        print(f"Warning: Could not list repo tree: {e}")
        # Fallback: try to find specific files
        for i in range(4):  # Try up to 4 files
            try:
                path = f"{folder_name}/params_layer_{i}.safetensors"
                hf_hub_download(
                    repo_id=repo_id,
                    filename=path,
                    force_download=False,  # Just check if it exists
                )
                params_files.append(path)
            except Exception:
                break

    if not params_files:
        raise ValueError(
            f"Could not find any params_layer_*.safetensors files in {folder_name}"
        )

    # Load and merge all weight files
    state_dict = {}

    for params_file in params_files:
        print(f"Loading {params_file}...")
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=params_file,
            force_download=force_download,
        )

        # Load weights from this file
        raw_state_dict = load_safetensors_weights(
            weights_path, device=device, dtype=cfg_dict.get("dtype")
        )

        print(f"  Keys in {params_file}: {list(raw_state_dict.keys())}")

        # Convert to SAELens naming convention
        # The raw state dict from Google has different naming
        weight_mapping = {
            "w_enc": "W_enc",
            "encoder_weight": "W_enc",
            "W_enc": "W_enc",
            "w_dec": "W_dec",
            "decoder_weight": "W_dec",
            "W_dec": "W_dec",
            "b_enc": "b_enc",
            "encoder_bias": "b_enc",
            "b_dec": "b_dec",
            "decoder_bias": "b_dec",
            "threshold": "threshold",
        }

        for raw_key, value in raw_state_dict.items():
            # Try to map the key
            mapped_key = weight_mapping.get(raw_key, raw_key)

            # If not mapped, keep the original name (might be SAELens format already)
            if mapped_key == raw_key and raw_key not in [
                "W_enc",
                "W_dec",
                "b_enc",
                "b_dec",
                "threshold",
            ]:
                # Check if the key contains common patterns
                if "enc" in raw_key.lower():
                    mapped_key = "W_enc"
                elif "dec" in raw_key.lower():
                    mapped_key = "W_dec"
                elif "bias_enc" in raw_key.lower():
                    mapped_key = "b_enc"
                elif "bias_dec" in raw_key.lower():
                    mapped_key = "b_dec"

            value = value.to(device=device)

            # For split files, concatenate along appropriate dimension
            if mapped_key in state_dict:
                # Weights need to be concatenated along the feature dimension
                if "W_enc" in mapped_key or "W_dec" in mapped_key:
                    state_dict[mapped_key] = torch.cat(
                        [state_dict[mapped_key], value], dim=0
                    )
                else:
                    state_dict[mapped_key] = torch.cat(
                        [state_dict[mapped_key], value], dim=0
                    )
            else:
                state_dict[mapped_key] = value

    # Ensure required keys exist
    required_keys = ["W_enc", "W_dec", "b_enc", "b_dec"]
    for key in required_keys:
        if key not in state_dict:
            print(f"Warning: Required key '{key}' not found in state_dict")

    print(f"Final state_dict keys: {list(state_dict.keys())}")
    for key, val in state_dict.items():
        print(f"  {key}: {val.shape}")

    return cfg_dict, state_dict, None
