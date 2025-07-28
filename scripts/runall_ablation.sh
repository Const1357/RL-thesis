#!/bin/bash

# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh alignment" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh alignment_penalty" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh alignment_margin" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh alignment_penalty_margin" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh penalty" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh penalty_margin" 
# ./scripts/gather5.sh "./scripts/pendulum_ablation_runner.sh margin" 

# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh alignment" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh alignment_penalty" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh alignment_margin" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh alignment_penalty_margin" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh penalty" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh penalty_margin" 
# ./scripts/gather5.sh "./scripts/acrobot_ablation_runner.sh margin" 

./scripts/gather5.sh "./scripts/pong_ablation_runner.sh" 
./scripts/gather5.sh "./scripts/pong_ablation_runner.sh alignment" 
./scripts/gather5.sh "./scripts/pong_ablation_runner.sh penalty" 
./scripts/gather5.sh "./scripts/pong_ablation_runner.sh margin" 
./scripts/gather5.sh "./scripts/pong_ablation_runner.sh penalty_margin" 

./scripts/gather5.sh "./scripts/runner.sh pong logits"