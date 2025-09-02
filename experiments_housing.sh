num_repetitions=(5 10 20)
nb_nodes_list=(4 10 20)
epsilons=(0.01 0.1 1 10)
graph_names=(empty expander complete cycle)
lrs=(0.01 0.1)

recompute_flag=""
for arg in "$@"; do
    if [[ "$arg" == "--recompute" ]]; then
        recompute_flag="--recompute"
    fi
done

# Preprocessing: count total configurations
total_configs=0
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for lr in "${lrs[@]}"; do
                    total_configs=$((total_configs + 1))
                done
            done
        done
    done
done

echo "Total configurations: $total_configs"

failed_configs=""
current_config=0

start_time=$(date +%s)
for num_repetition in "${num_repetitions[@]}"; do
    for nb_nodes in "${nb_nodes_list[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for graph_name in "${graph_names[@]}"; do
                for lr in "${lrs[@]}"; do
                    current_config=$((current_config + 1))
                    echo "Running configuration $current_config / $total_configs"
                    cmd="python simulations/housing.py --nb_nodes $nb_nodes --lr $lr --num_repetition $num_repetition --nb_batches 16 --epsilon $epsilon --graph_name $graph_name --use_optimals $recompute_flag"  
                    echo "$cmd":
                    if ! $cmd; then
                        failed_configs+=("$cmd"\n)
                    fi
                done
            done
        done
    done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "All experiments took ${hours}h ${minutes}m ${seconds}s."
echo Failed configs: "$failed_configs"