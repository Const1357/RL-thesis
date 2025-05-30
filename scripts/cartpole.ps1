param (
    [string]$type = "logits"  # default value
)

# run example: .\scripts\cartpole.ps1 -type logits

if ($type -eq "logits") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_Logits.yaml 
}
elseif ($type -eq "entropy") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_Entropy.yaml
}
elseif ($type -eq "gnn") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN.yaml
}
elseif ($type -eq "gnnk") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_K.yaml
}
elseif ($type -eq "gn3") {
    python main.py --common configs/cartpole/cartpole_common.yaml --config configs/cartpole/cartpole_GNN_N.yaml
}
else {
    Write-Host "Unknown mode: $type"
}