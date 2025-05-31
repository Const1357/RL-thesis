param (
    [string]$type = "logits"  # default value
)

# run example: .\scripts\cartpole.ps1 -type logits

if ($type -eq "logits") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_Logits.yaml 
}
if ($type -eq "logits_scheduler") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_Logits_Scheduler.yaml 
}
elseif ($type -eq "entropy") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_Logits_Entropy.yaml
}
elseif ($type -eq "gnn") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN.yaml
}
elseif ($type -eq "gnn_frozen") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_frozen.yaml
}
elseif ($type -eq "gnnk") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_K.yaml
}
elseif ($type -eq "gnnk_frozen") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_K_frozen.yaml
}
elseif ($type -eq "gn3") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_N.yaml
}
else {
    Write-Host "Unknown mode: $type"
}