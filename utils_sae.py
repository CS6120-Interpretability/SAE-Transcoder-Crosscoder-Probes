# %%
import re
from typing import List, Sequence, Union

from sae_lens import SAE

BASEPATH = '../SAE-Probing'
# we use this to point towards a directory where we host model activations

def sanitize_sae_id(sae_id: str) -> str:
    """
    Convert an SAE identifier into a filesystem-friendly string.

    Keeps alphanumerics, dots, dashes, and underscores while replacing other
    characters with hyphens.
    """
    sanitized = []
    for char in str(sae_id):
        if char.isalnum() or char in "._-":
            sanitized.append(char)
        else:
            sanitized.append("-")
    return "".join(sanitized)

def build_sae_description(
    dataset: str,
    sae_id: Union[str, Sequence[str]],
    layer: Union[int, str, None] = None,
) -> str:
    """
    Build the dataset/SAE description string used in activation and probe paths.

    Preserve the legacy naming scheme by inferring width/L0 when present, but
    otherwise fall back to a sanitized SAE id string.
    """
    if isinstance(sae_id, (list, tuple)) and len(sae_id) >= 4:
        try:
            name = '_'.join(str(sae_id[2]).split('/')[0].split('_')[1:])
            l0 = round(float(sae_id[3]))
            return f"{dataset}_{name}_{l0}"
        except Exception:
            pass

    sae_id_str = str(sae_id)
    if sae_id_str.startswith("l") and "r_" in sae_id_str and sae_id_str[1:2].isdigit():
        return f"{dataset}_{sae_id_str}"

    parts = sae_id_str.split("/")
    width = next((p for p in parts if p.startswith("width_")), None)
    l0 = next((p for p in parts if "average_l0_" in p), None)
    if width and l0:
        layer_label = layer if layer is not None else "custom"
        return f"{dataset}_{layer_label}_{width}_{l0}"

    if layer is None:
        return f"{dataset}_{sanitize_sae_id(sae_id_str)}"
    return f"{dataset}_{layer}_{sanitize_sae_id(sae_id_str)}"

def get_sae_hook_name(sae: SAE) -> str:
    """
    Extract the hook name from an SAE, supporting both dict- and object-based configs.
    """
    if hasattr(sae, "cfg") and hasattr(sae.cfg, "hook_name"):
        return sae.cfg.hook_name
    if hasattr(sae, "cfg_dict") and "hook_name" in sae.cfg_dict:
        return sae.cfg_dict["hook_name"]
    if hasattr(sae, "cfg") and isinstance(sae.cfg, dict) and "hook_name" in sae.cfg:
        return sae.cfg["hook_name"]
    raise ValueError("Could not determine hook_name from SAE config.")

def infer_layer_label(
    layer: Union[int, str, None],
    hook_name: Union[str, Sequence[str], None],
) -> Union[int, str]:
    """
    Choose a stable layer label for file naming, falling back to hook_name when
    the layer is unknown or non-standard.
    """
    if layer is not None:
        return layer
    if isinstance(hook_name, (list, tuple)):
        layers = []
        for name in hook_name:
            if isinstance(name, str) and name.startswith("blocks."):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layers.append(parts[1])
        if layers:
            return f"layers_{'_'.join(layers)}"
        return "multi"

    if isinstance(hook_name, str) and hook_name.startswith("blocks."):
        parts = hook_name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
    return hook_name or "custom"

def parse_layers_from_sae_id(sae_id: str) -> List[int]:
    """Extract layer indices from SAE ids like 'layer_7_13_17_22_width_262k_l0_medium'."""
    match = re.search(r"layer_(\d+(?:_\d+)*)", sae_id)
    if not match:
        return []
    return [int(layer) for layer in match.group(1).split("_")]
