# MFDL: Matrix Factorization for Decentralized Learning

Matrix Factorization for Decentralized Learning (MFDL) is a Python package for simulating and analyzing decentralized learning algorithms using matrix factorization techniques.

## Features
- Optimal workload experiments
- Housing dataset experiments with graph-based correlations
- Accounting experiments
- Visualization tools for simulation results

## Prerequisites
- Python 3.12 (only tested version)
- Recommended: virtual environment (`python3.12 -m venv venv`)

## Installation

1. Install dependencies:
    ```bash
    pip install -r requirement.txt
    ```

## Usage
By default, all workload matrices are cached in the `cache/` directory to save computation time. Pre-computed matrices are included with this submission.
This allows to skip expensive steps, especially for housing and large graphs.
To regenerate all workloads from scratch, delete the `cache/` directory and rerun the experiments.
Be warned this may take time and can be expensive in terms of memory.


### 1. Accounting Experiments

Run:
```bash
python simulations/muffliato_accounting.py
```

### 2. Housing Experiments
Note this is the most expensive and longest computation here!
Step 1 should require around 200GB of RAM (creating the workloads for peertube and ego is expensive, even if we optimize it to the maximum by caching repetitive parts), and step 2 requires simulating decentralized learning in multiple settings, which may take time. 
It is recommended to follow the order described below.
For the paper, we used a 256 cores machine with 512GB of RAM, and required around 10 hours to perform all graphs.


**Step 1: Pre-cache optimal correlations**

Run for each graph (`florentine`, `ego`, `peertube` and `misskey`):
```bash
./experiments_housing.sh --threads=1 --graph=<graph_name> --pre_cache
```
Cached workloads will be stored under `cache/`.

**Step 2: Run simulations**
This steps is the longest, and may take a few hours depending on how many thread you use.
Set environment variables for optimal parallelism:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./experiments_housing.sh --threads=15 --graph=<graph_name>
```
*Adjust `--threads` and `--graph` as needed. As a general rule of thumbs, each process will generate 8 threads, so have threads = number of cores//8*

**Step 3: Visualize results**

The privacy-utility tradeoffs are displayed using:
```bash
python simulations/tradeoff_plotter.py --dataset <dataset_name>
```
with `dataset_name` being either `housing` or `femnist`.
This will generate the corresponding figures under `figures/`.

### 3. Femnist Experiments
Follow the same steps as housing, but replace `experiments_housing.sh` with `experiments_femnist.sh`, and only use the graphs `florentine` and `ego`.

### 4. Optimal Workload Experiments

Run:
```bash
python simulations/factorization_experiments.py
```

Resulting figures are stored under `figures/factorization_simulation/`. 


## Datasets

- **Facebook Ego Graph:**  
  Download from [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) and place under `graphs/facebook/` (ensure at least `414.edges` is present).

## Troubleshooting

- If you encounter issues with parallelism or performance, ensure the environment variables above are set before running simulations.
- For missing datasets, verify the correct file paths and download locations.
- You may need to create empty directories `results/` or `figures/`.

## License

See `LICENSE` for details.

## Contact

For questions or contributions, please open an issue or pull request.