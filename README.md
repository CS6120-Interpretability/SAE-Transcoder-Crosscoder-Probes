# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from our paper [*Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*](https://arxiv.org/pdf/2502.16681). The workflow of our code involves three primary stages. Each part should be mostly executable independently from artifacts we make available:

1. **Generating Model and SAE Activations:**
   - Model activations for probing datasets are generated in `generate_model_activations.py`
   - SAE activations are generated in `generate_sae_activations.py`. Because of CUDA memory leakage, we rerun the script for every SAE, we do this in `save_sae_acts_and_train_probes.sh`, which should work if you just run it.
   - OOD regime activations are specifically generated in `plot_ood.ipynb`.
   - Mutli-token activations are specifically generated in `generate_model_and_sae_multi_token_acts.py`. Caution: this will take up a lot of memory (~1TB).

2. **Training Probes:**
   - Baseline probes are trained using `run_baselines.py`. This script also includes additional functions for OOD experiments related to probe pruning and latent interpretability (see Sections 4.1 and 4.2 of the paper).
   - SAE probes are trained using `train_sae_probes.py`. Sklearn regression is most efficient when run in a single thread, and then many of those threads can be run in parallel. We do this in `save_sae_acts_and_train_probes.sh`.
   - Multi token SAE probes and baseline probes are trained using `run_multi_token_acts.py`.
   - Combining all results into csvs after they are done is done with `combine_results.py`.

3. **Visualizing Results:**
   - Standard condition plots: `plot_normal.ipynb`
   - Data scarcity, class imbalance, and corrupted data regimes: `plot_combined.ipynb`
   - OOD plots: `plot_ood.ipynb`
   - Llama-3.1-8B results replication: `plot_llama.ipynb`
   - GLUE CoLA and AIMade investigations (Sections 4.3.1 and 4.3.2): `dataset_investigations/`
   - AI vs. human final token plots: `ai_vs_humanmade_plot.py`
   - SAE architectural improvements (Section 6): `sae_improvement.ipynb`
   - Multi token: `plot_multi_token.py`
   - K vs. AUC plot broken down by dataset (in appendix): `k_vs_auc_plot.py` 
   
Note that these should all be runnable as is from the results data in the repo.

### Custom Models and SAEs
We now support arbitrary TransformerLens models and SAE Lens releases, not just the three paper models. The key idea is to keep file paths keyed off a `model_name` (path-safe string) while allowing a separate model id for loading and explicit SAE ids/releases for SAE Lens.

**Model activations (any TransformerLens model):**
```
python generate_model_activations.py --model_name my-model-name --model_id meta-llama/Llama-3.1-8B \
  --layers 0 10 20 --no_embed
```
Optional flags: `--hook_names` (full hook names), `--layers`, `--resid_hook`, `--num_default_layers`, `--no_embed`.

**SAE activations (any SAE Lens release/id):**
```
python generate_sae_activations.py --model_name my-model-name --setting normal \
  --sae_release <sae_release> --sae_ids <sae_id> --hook_name blocks.20.hook_resid_post
```
Optional flags: `--layers` (override layer list when model defaults are unknown).
For multi-layer crosscoders, pass `--hook_names` (one per layer) and, if needed,
`--sae_converter module:function` to use a custom loader.

**SAE probe training (custom layers):**
```
python train_sae_probes.py --model_name my-model-name --reg_type l1 --setting normal \
  --target_sae_id <sae_id> --layers 20
```

**Multi-token activations (custom models/SAEs):**
```
python generate_model_and_sae_multi_token_acts.py --model_name my-model-name --model_id meta-llama/Llama-3.1-8B \
  --hook_name blocks.20.hook_resid_post --sae_release <sae_release> --sae_ids <sae_id>
```
If the SAE config exposes a hook name, you can omit `--hook_name`; otherwise pass it explicitly.

**Multi-token probe training (custom hooks/SAEs):**
```
python run_multi_token_acts.py --model_name my-model-name --max_seq_len 256 --layer 20 \
  --hook_name blocks.20.hook_resid_post --sae_id <sae_id> --to_run_list sae_aggregated
```

**Notes on consolidated probing filenames:**
When `run_multi_token_acts.py` is called with an explicit `--sae_id`, the
consolidated probing outputs use the sanitized SAE id in the filename (for
example, `.../<dataset>_<layer>_<sae_id>_mean.pkl`). If you omit `--sae_id` and
use the default Gemma-2-9B settings, the legacy `width16k_l0{L0}` naming is
preserved. The plotting script (`plot_multi_token.py`) now detects either
format and selects a primary SAE label automatically when computing summary
statistics.

**Legacy SAE helpers:**
Model-specific SAE utilities used in the original paper runs now live in
`legacy_sae.py`. Import from `legacy_sae.py` when you need Gemma/Llama SAE id
lookups or pretrained release names. New code paths should prefer the generic
helpers in `utils_sae.py` and pass explicit SAE ids/releases when needed.

### Datasets
- **Raw Text Datasets:** Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0). Note that datasets 161-163 are modified from their source. An error in our formatting reframes them as differentiating between news headlines and code samples. 
- **Model Activations:** Also stored on Dropbox (Note: Files are large).

## Requirements
We recommend you create a new python venv named probing and install required packages with pip:
```
python -m venv probing
source probing/bin/activate
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort
```
Let us know if anything does not work with this environment!


For any questions or clarifications, please open an issue or reach out to us!
