#!/bin/bash

# num_repetitions=(5 10)
# nb_nodes_list=(10 20 50 100)
# epsilons=(0.01 0.05 0.1 0.5 1 10)
# lrs=(0.1)
# graph_names=(empty expander complete cycle erdos chain)
# seeds=(421 422 423 424 425 426 427 428 429 430)

num_repetitions=(20)
# mu_list=(1.0 0.1 10.0 0.5 2.0 0.2 5.0)
mu_list=(0.1 0.5 1.0 2.0 3.0 4.0 5.0 7.0 8.0 9.0 10.0 15.0 20.0) # Only use this for florentine?
# nb_nodes_list=(15)
# graph_names=(florentine)

# nb_nodes_list=(148)
# graph_names=(ego)

# nb_nodes_list=(271)
# graph_names=("peertube (connex component)")

# lrs=(0.1)
# lrs=(1 0.01 0.1 0.001 0.05 0.005 0.02 2 5 10 25 50)

# micro_batches_per_iteration=(10 5 1)
micro_batches_per_iteration=(1)


seeds=(421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440)
# seeds=(421)


max_jobs=1
recompute_flag=""
pre_cache_flag=""
run_with="local"
hyperparameter_flag="--use_optimals"
skip_confirmation=false
for arg in "$@"; do
    if [[ "$arg" == "--recompute" ]]; then
        recompute_flag="--recompute"
    elif [[ "$arg" == "--pre_cache" ]]; then
        pre_cache_flag="--pre_cache"
        max_jobs=1
    elif [[ "$arg" == "--graph=florentine" ]]; then
        nb_nodes_list=(15)
        graph_names=("florentine")
        lrs=(5.0) # Florentine
    elif [[ "$arg" == "--graph=ego" ]]; then
        nb_nodes_list=(148)
        graph_names=("ego")
        lrs=(2.0) # Ego
    elif [[ "$arg" == "--graph=peertube" ]]; then
        nb_nodes_list=(271)
        graph_names=("peertube (connex component)")
    elif [[ "$arg" == --threads=* ]]; then
        max_jobs="${arg#--threads=}"
    elif [[ "$arg" == --run_with=slurm ]]; then
        run_with="slurm"
    elif [[ "$arg" == "--hyperparameters" ]]; then
        hyperparameter_flag="--hyperparameters"
        mu_list=(1)
        seeds=(421)
        lrs=(1 0.01 0.1 0.001 0.05 0.005 0.02 2 5 10 25 50)
    elif [[ "$arg" == "-y" ]]; then
        skip_confirmation=true
    else
        echo "Error: Unrecognized argument '$arg'" >&2
        exit 1
    fi
done

# Preprocessing: count total configurations
total_configs=0
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for mu in "${mu_list[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for nb_micro_batches in "${micro_batches_per_iteration[@]}"; do
                            total_configs=$((total_configs + 1))
                        done
                    done
                done
            done
        done
    done
done

echo "num_repetitions: ${num_repetitions[@]}"
echo "nb_nodes_list: ${nb_nodes_list[@]}"
echo "mu_list: ${mu_list[@]}"
echo "graph_names: ${graph_names[@]}"
echo "lrs: ${lrs[@]}"
echo "micro_batches_per_iteration: ${micro_batches_per_iteration[@]}"
echo "seeds: ${seeds[@]}"
echo "max_jobs: $max_jobs"
echo "run_with: $run_with"
echo "hyperparameter_flag: $hyperparameter_flag"
echo "recompute_flag: $recompute_flag"
echo "pre_cache_flag: $pre_cache_flag"
echo "Total configurations: $total_configs"
echo "Proceed with these parameters? (y/n)"
if ! $skip_confirmation; then
    read -r answer
    if [[ "$answer" != "y" ]]; then
        echo "Aborted by user."
        exit 0
    fi
fi


current_config=0

job_count=0
pids=()

# Trap SIGINT (Ctrl+C) and kill all child process groups
trap 'echo "Killing all child jobs..."; for pgid in "${pids[@]}"; do kill -- -$pgid 2>/dev/null; done; exit 1' SIGINT



start_time=$(date +%s)
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for mu in "${mu_list[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for nb_micro_batches in "${micro_batches_per_iteration[@]}"; do
                    for lr in "${lrs[@]}"; do
                        for seed in "${seeds[@]}"; do
                            current_config=$((current_config + 1))
                            echo "Running configuration $current_config / $total_configs"
                            cmd=(python -u simulations/decentralized_simulation.py --nb_nodes $nb_nodes --lr $lr --num_repetition $num_repetition --nb_batches 16 --mu $mu --graph_name "$graph_name" --use_optimals $hyperparameter_flag $recompute_flag $pre_cache_flag --dataloader_seed $seed --dataset femnist --nb_micro_batches $nb_micro_batches)
                            if [[ "$run_with" == "slurm" ]]; then
                                slurm_cmd=(sbatch ./jean_zay_launch.slurm "\"${cmd[@]}\"")
                                echo "${slurm_cmd[@]}"
                                eval "${slurm_cmd[@]}"
                            else
                                echo "${cmd[@]}"
                                setsid "${cmd[@]}" &
                                pid=$!
                                pids+=($pid)
                                job_count=$((job_count + 1))
                                if (( job_count >= max_jobs )); then
                                    wait -n
                                    job_count=$((job_count - 1))
                                fi
                            fi
                        done
                    done
                done
            done
        done
    done
done
wait # Wait for all remaining jobs


end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "All experiments took ${hours}h ${minutes}m ${seconds}s."