param (
    [string]$command
)

for ($i = 1; $i -le 5; $i++) {
    Write-Host "Run #$i"
    Invoke-Expression $command
}

# run example: .\scripts\gather5 ".\scripts\runner cartpole logits noise"