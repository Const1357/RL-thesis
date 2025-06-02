param($env, $type, $mod)

python main.py --common configs/$env/${env}_common.yaml --config configs/$env/${env}_${type}_${mod}.yaml

# example run:
# .\scripts\runner cartpole logits
# .\scripts\runner cartpole logits noise
# .\scripts\runner cartpole logits entropy
# .\scripts\runner cartpole logits noise_entropy