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
By default, all workload matrices are cached in the `cache/` directory to save computation time. Pre-computed matrices are included with this submission. To regenerate workloads from scratch, delete the `cache/` directory and rerun the experiments.


### 1. Accounting Experiments

Run:
```bash
python simulations/muffliato_accounting.py
```

### 2. Housing Experiments
Note this is the most expensive computation here!
Step 1 should require around 200GB of RAM (creating the workloads for peertube and ego is expensive, even if we optimize it to the maximum by caching repetitive parts), and step 2 requires simulating decentralized learning in multiple settings, which may take time. 
It is recommended to 
For the paper, we used a 256 cores machine with 512GB of RAM.


**Step 1: Pre-cache optimal correlations**

Run for each graph (`florentine`, `ego`, `peertube`):
```bash
./experiments_housing.sh --threads=1 --graph=<graph_name> --pre_cache
```
Cached workloads will be stored under `cache/`.

**Step 2: Run simulations**

Set environment variables for optimal parallelism:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./experiments_housing.sh --threads=15 --graph=<graph_name>
```
*Adjust `--threads` and `--graph` as needed. As a general rule of thumbs, each process will generate 8 threads, so have threads = number of cores//8*

**Step 3: Visualize results**

```bash
python simulations/housing_plotter.py
```
> Figures will be under `figures/housing/`.
`epsilon` in their description is somewhat poorly named, and corresponds to $\frac{1}{\sigma}$.

Run `simulations/epsilon_computation.py` to have the corresponding $\varepsilon$ used in the paper.

### 3. Optimal Workload Experiments

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

## License

See `LICENSE` for details.

## Contact

For questions or contributions, please open an issue or pull request.