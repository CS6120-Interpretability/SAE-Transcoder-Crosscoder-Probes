# ---- Generate SAE activations ----

MODEL_NAME="gemma-2-9b"
DEVICE="cuda:1"
OMP_THREADS="1"

# Optional overrides for custom models/SAEs:
# SAE_RELEASE=""
# SAE_IDS=""
# HOOK_NAME=""
# LAYERS=""

GEN_NORMAL_ITERS=100
GEN_OTHER_ITERS=10
PROBE_ITERS=20

for i in $(seq 1 ${GEN_NORMAL_ITERS})
do
    python3 generate_sae_activations.py \
        --model_name "${MODEL_NAME}" \
        --setting normal \
        --device "${DEVICE}" \
        ${SAE_RELEASE:+--sae_release "${SAE_RELEASE}"} \
        ${SAE_IDS:+--sae_ids ${SAE_IDS}} \
        ${HOOK_NAME:+--hook_name "${HOOK_NAME}"} \
        ${LAYERS:+--layers ${LAYERS}}
done

# ---- Train SAE probes ----

for i in $(seq 1 ${PROBE_ITERS})
do
    OMP_NUM_THREADS="${OMP_THREADS}" python3 train_sae_probes.py \
        --model_name "${MODEL_NAME}" \
        --setting normal \
        --reg_type l1 \
        --randomize_order \
        ${LAYERS:+--layers ${LAYERS}} &
done
