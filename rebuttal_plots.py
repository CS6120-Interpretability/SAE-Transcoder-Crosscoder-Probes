"""
Generate supplemental figures used in the rebuttal analysis.

This script compares SAE performance across layers (largest L0 per layer) and
contrasts old vs. new mean-difference feature selection runs, emitting figures
into ``rebuttal_plots/``.
"""
# %%
import pandas as pd
all_data = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv")
# %%

from legacy_sae import get_sae_layers_extra, get_gemma_2_9b_sae_ids

# Get all layers
layers = get_sae_layers_extra("gemma-2-9b")

# For each layer, find SAE with largest l0
largest_l0_saes = {}
for layer in layers:
    sae_ids = get_gemma_2_9b_sae_ids(layer)
    
    # Get l0 values for each SAE
    l0s = [int(sae_id.split("/")[2].split("_")[-1]) for sae_id in sae_ids]
    
    # Find SAE with largest l0
    max_l0_idx = l0s.index(max(l0s))
    largest_l0_saes[layer] = sae_ids[max_l0_idx]

print("SAEs with largest l0 for each layer:")
for layer, sae_id in largest_l0_saes.items():
    print(f"Layer {layer}: {sae_id}")


# Filter data to only include largest l0 SAEs
filtered_data = pd.DataFrame()
for layer, sae_id in largest_l0_saes.items():
    layer_data = all_data[all_data["sae_id"] == sae_id]
    filtered_data = pd.concat([filtered_data, layer_data])

# Calculate average test_auc for each layer
layer_aucs = filtered_data.pivot_table(index="k", columns="layer", values="test_auc", aggfunc="mean")

# Create line plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for layer in layer_aucs.columns:
    plt.plot(layer_aucs.index, layer_aucs[layer], marker='o', label=f'Layer {layer}')
plt.title("Average Test AUC by Layer (Largest L0 SAEs)")
plt.xlabel("k")
plt.ylabel("Average Test AUC")
plt.xscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0.8, 1)
plt.legend()
plt.tight_layout()
plt.savefig("rebuttal_plots/comparing_sae_test_auc_by_layer.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %%

import pandas as pd

new_results = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv")
old_results = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics_old.csv")
# %%

# Create a merged dataframe matching on k, dataset, and layer
merged_df = pd.merge(
    old_results[["k", "dataset", "test_auc", "sae_id"]],
    new_results[["k", "dataset", "test_auc", "sae_id"]], 
    on=["k", "dataset", "sae_id"],
    suffixes=("_old", "_new")
)

# %%

# Calculate average test_auc for each dataset
avg_by_dataset = merged_df.groupby(["k", "sae_id"]).agg({
    "test_auc_old": "mean",
    "test_auc_new": "mean"
}).reset_index()

# Print the results
print("\nAverage test AUC across datasets:")
print(avg_by_dataset)

# %%
import matplotlib.pyplot as plt
from natsort import natsorted

# Get unique SAE IDs
sae_ids = natsorted(avg_by_dataset["sae_id"].unique())

# Move first 6 SAEs to the end
sae_ids = sae_ids[6:] + sae_ids[:6]

# Create a figure with subplots for each SAE ID
n_plots = len(sae_ids)
n_cols = 5
n_rows = (n_plots + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
axes = axes.flatten()

for i, sae_id in enumerate(sae_ids):
    # Get data for this SAE ID
    sae_data = avg_by_dataset[avg_by_dataset["sae_id"] == sae_id]
    
    # Plot both lines
    axes[i].plot(sae_data["k"], sae_data["test_auc_old"], label="Choosing By Mean Diff", marker='o')
    axes[i].plot(sae_data["k"], sae_data["test_auc_new"], label="Normalizing And Then Choosing By Mean Diff", marker='o')
    
    # Customize plot
    axes[i].set_title(sae_id)
    axes[i].set_xlabel("k")
    axes[i].set_ylabel("Test AUC")
    # axes[i].legend()
    axes[i].grid(True)
    axes[i].set_xscale('log')

# Remove any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Choosing by Mean Diff (Blue) vs Normalizing And Then Choosing by Mean Diff (Orange)", y=1.02, fontsize=22)

plt.tight_layout()
plt.savefig("rebuttal_plots/comparing_new_and_old_mean_diff_auc.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %%
