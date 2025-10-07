# num_repetitions=(5 10)
# nb_nodes_list=(10 20 50 100)
# epsilons=(0.01 0.05 0.1 0.5 1 10)
# lrs=(0.1)
# graph_names=(empty expander complete cycle erdos chain)
# seeds=(421 422 423 424 425 426 427 428 429 430)

num_repetitions=(20)
mu=(1 0.1 10 0.5 2 0.2 5)

# nb_nodes_list=(15)
# graph_names=(florentine)

# nb_nodes_list=(148)
# graph_names=(ego)

# nb_nodes_list=(271)
# graph_names=("peertube (connex component)")

lrs=(0.1)

seeds=(421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440)


max_jobs=1
recompute_flag=""
pre_cache_flag=""
for arg in "$@"; do
    if [[ "$arg" == "--recompute" ]]; then
        recompute_flag="--recompute"
    elif [[ "$arg" == "--pre_cache" ]]; then
        pre_cache_flag="--pre_cache"
        max_jobs=1
    elif [[ "$arg" == "--graph=florentine" ]]; then
        nb_nodes_list=(15)
        graph_names=("florentine")
    elif [[ "$arg" == "--graph=ego" ]]; then
        nb_nodes_list=(148)
        graph_names=("ego")
    elif [[ "$arg" == "--graph=peertube" ]]; then
        nb_nodes_list=(271)
        graph_names=("peertube (connex component)")
    elif [[ "$arg" == --threads=* ]]; then
        max_jobs="${arg#--threads=}"
    else
        echo "Error: Unrecognized argument '$arg'" >&2
        exit 1
    fi
done

# Preprocessing: count total configurations
total_configs=0
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for mu in "${mu[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        total_configs=$((total_configs + 1))
                    done
                done
            done
        done
    done
done

echo "Total configurations: $total_configs"

current_config=0

job_count=0
pids=()

# Trap SIGINT (Ctrl+C) and kill all child process groups
trap 'echo "Killing all child jobs..."; for pgid in "${pids[@]}"; do kill -- -$pgid 2>/dev/null; done; exit 1' SIGINT



start_time=$(date +%s)
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for mu in "${mu[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        current_config=$((current_config + 1))
                        echo "Running configuration $current_config / $total_configs"
                        cmd=(python simulations/decentralized_simulation.py --nb_nodes $nb_nodes --lr $lr --num_repetition $num_repetition --nb_batches 16 --mu $mu --graph_name "$graph_name" --use_optimals $recompute_flag $pre_cache_flag --dataloader_seed $seed --dataset femnist)
                        echo "${cmd[@]}":
                        setsid "${cmd[@]}" &
                        pid=$!
                        pids+=($pid)
                        job_count=$((job_count + 1))
                        if (( job_count >= max_jobs )); then
                            wait -n
                            job_count=$((job_count - 1))
                        fi
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