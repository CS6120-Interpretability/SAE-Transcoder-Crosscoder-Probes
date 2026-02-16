# %%
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import MODEL_CONFIGS, snapshot_download
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import load_dictionary_learning_sae
from sae_bench.sae_bench_utils import general_utils


def get_all_hf_repo_autoencoders(
    repo_id: str, download_location: str = "downloaded_saes"
) -> list[str]:
    """
    Download SAE configs from a HuggingFace repo and return relative locations
    for each SAE contained within the snapshot.
    """
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))
    config_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.json"],
        local_dir=download_location,
        force_download=False,
    )

    config_locations = []

    for root, _, files in os.walk(config_dir):
        for file in files:
            if file.endswith(".json"):
                config_locations.append(os.path.join(root, file))

    repo_locations = []

    for config in config_locations:
        if "config.json" in config:
            repo_location = config.split(f"{download_location}/")[1].split("/config.json")[
                0
            ]
            repo_locations.append(repo_location)

    return repo_locations


def get_gemma_2_2b_sae_ids(layer: int) -> List[Tuple[str, str, str, float]]:
    """List all Gemma-2-2B SAE identifiers discovered in the configured repos."""
    assert layer == 12

    repo_ids = ["canrager/saebench_gemma-2-2b_width-2pow14_date-0107", "adamkarvonen/temp"]
    model_name = "gemma-2-2b"
    exclude_keywords = ["checkpoints"]
    include_keywords = []

    all_sae_locations = []
    for repo_id in repo_ids:
        if "adamkarvonen" in repo_id:
            include_keywords_local = ["old_relu_google_gemma-2-2b_standard"]
        else:
            include_keywords_local = include_keywords

        print(f"\n\n\nEvaluating {model_name} with {repo_id}\n\n\n")

        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]

        torch_dtype = torch.float32

        sae_locations = get_all_hf_repo_autoencoders(repo_id)

        sae_locations = general_utils.filter_keywords(
            sae_locations, exclude_keywords=exclude_keywords, include_keywords=include_keywords_local
        )


        cfg_paths = [f"downloaded_saes/{repo_id.replace('/', '_')}/{loc}/eval_results.json" for loc in sae_locations]
        l0s = [json.load(open(cfg_path))["l0"] for cfg_path in cfg_paths]
        all_sae_locations.extend([(repo_id, model_name, loc, l0) for loc, l0 in zip(sae_locations, l0s)])

    return all_sae_locations

def get_gemma_2_2b_sae_ids_largest_l0(layer: int) -> List[Tuple[str, str, str]]:
    """
    Select one SAE per dictionary-learning run, picking the checkpoint whose L0
    is closest to 200 to keep comparisons manageable.
    """
    assert layer == 12

    all_sae_locations = get_gemma_2_2b_sae_ids(layer)

    # Group SAE locations by their base path (everything except the last part)
    base_paths = defaultdict(list)
    for repo_id, model_name, loc, l0 in all_sae_locations:
        # Split path and group by everything except last component
        path_parts = loc.split('/')
        base = '/'.join(path_parts[:-1])
        base_paths[base].append((repo_id, model_name, loc, l0))

    filtered_locations = []
    for base, locations in base_paths.items():
        if len(locations) == 1:
            filtered_locations.extend(locations)
            continue
            
        target_l0 = 200
        best_distance_to_target_l0 = float("inf")
        best_loc = None
        for repo_id, model_name, loc, l0 in locations:
            distance_to_target_l0 = abs(l0 - target_l0)
            if distance_to_target_l0 < best_distance_to_target_l0:
                best_distance_to_target_l0 = distance_to_target_l0
                best_loc = (repo_id, model_name, loc)

        if best_loc:
            filtered_locations.append(best_loc)

    return filtered_locations

def load_gemma_2_2b_sae(sae_location: Tuple[str, str, str]):
    """Load a Gemma-2-2B SAE from disk via sae-bench helpers."""
      
    repo_id, model_name, sae_location, l0 = sae_location

    return load_dictionary_learning_sae(
        repo_id=repo_id,
        location=sae_location,
        layer=None,
        model_name=model_name,
        device="cpu",
        dtype=torch.float32,
    )
