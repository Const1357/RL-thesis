$envs  = "cartpole","pendulum"
$types = "logits","GNN","GNN_K"
$mods  = "","noise","entropy","noise_entropy"

foreach ($env in $envs) {
    foreach ($type in $types) {
        foreach ($mod in $mods) {
            & .\scripts\gather5 ".\scripts\runner $env $type $mod"
        }
    }
}
