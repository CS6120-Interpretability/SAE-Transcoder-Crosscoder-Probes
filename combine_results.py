# %%
import glob
import os
import pickle
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
def process_metrics(file: str, model_name: str) -> Optional[List[dict]]:
    """
    Load a pickled metrics file and normalize SAE metadata so downstream CSVs
    are comparable across model families.

    Args:
        file: Path to a ``.pkl`` metrics file produced by probe training.
        model_name: Name of the model whose metrics are being loaded. Used to
            harmonize SAE identifiers for Gemma-2-2B runs.

    Returns:
        The unpickled metrics list if loading succeeds, otherwise ``None`` to
        flag a bad file.
    """
    with open(file, "rb") as f:
        try:
            metrics = pickle.load(f)
            if model_name == "gemma-2-2b":
                for metric in metrics:
                    sae_id = metric["sae_id"]
                    name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
                    l0 = sae_id[3]
                    rounded_l0 = round(float(l0))
                    metric["sae_id"] = f"{name}"
                    metric["sae_l0"] = rounded_l0
            return metrics
        except Exception:
            return None

def process_files(files: List[str], model_name: str) -> Tuple[List[list], List[str]]:
    """
    Iterate through a list of metrics pickles, recording which files failed to
    load so the caller can sanity-check missing data.

    Args:
        files: List of metrics file paths to unpickle.
        model_name: Model identifier passed through to ``process_metrics``.

    Returns:
        A tuple of ``(loaded_metrics, bad_files)``.
    """
    all_metrics: List[list] = []
    bad_files: List[str] = []
    
    file_iterator = tqdm(files)
    
    for file in file_iterator:
        metrics = process_metrics(file, model_name)
        if metrics:
            all_metrics.append(metrics)
        else:
            bad_files.append(file)
    
    return all_metrics, bad_files

def extract_sae_features(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Derive SAE width and L0 metadata columns for Gemma-2-9B runs.

    Args:
        df: Aggregated metrics dataframe.
        model_name: Model identifier; currently only adds features for
            ``gemma-2-9b``.

    Returns:
        The same dataframe with SAE metadata columns added when applicable.
    """
    if model_name == "gemma-2-9b":
        df.loc[:, "sae_width"] = df["sae_id"].apply(lambda x: x.split("/")[1].split("_")[1])
        df.loc[:, "sae_l0"] = df["sae_id"].apply(lambda x: int(x.split("/")[2].split("_")[2]))
    return df

def process_setting(setting: str, model_name: str) -> None:
    """
    Collect all probe metrics for a model/setting combination into a single
    CSV, asserting that no corrupt pickles were encountered.

    Args:
        setting: Experiment regime (``normal``, ``scarcity``, ``class_imbalance``,
            ``label_noise``, or ``OOD``).
        model_name: Model identifier matching the directory layout.
    """
    print(f"Processing {setting} setting for {model_name}...")
    
    # Create output directory
    output_dir = f"results/sae_probes_{model_name}/{setting}_setting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file pattern based on setting
    file_pattern = f"data/sae_probes_{model_name}/{setting}_setting/*.pkl"
    
    # Process files
    files = glob.glob(file_pattern)
    print(file_pattern)
    print(len(files))
    if len(files) == 0:
        return
    
    all_metrics, bad_files = process_files(files, model_name)
    assert len(bad_files) == 0, f"Found {len(bad_files)} bad files in {setting} setting"
    
    # Create dataframe
    df = pd.DataFrame([item for sublist in all_metrics for item in sublist])
    
    # Save to CSV
    df.to_csv(f"{output_dir}/all_metrics.csv", index=False)
        
    # Print dataset length
    print(f"Total records in {setting} setting: {len(df)}")
        
    

# %%

for setting in ["normal", "scarcity", "class_imbalance", "label_noise", "OOD"]:
    for model_name in ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]:
        process_setting(setting, model_name)
# %%
