# MFDL: Matrix Factorization for Decentralized Learning

Matrix Factorization for Decentralized Learning (MFDL) is a Python package for simulating and analyzing decentralized learning algorithms using matrix factorization techniques.

This repository was developed as part of the work [Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](https://arxiv.org/abs/2510.17480), accepted at ICLR 2026.

## Attribution
If you use this code in your research, please cite the original paper:

```
@inproceedings{Bellet2026unified,
  title={Unified {{Privacy Guarantees}} for {{Decentralized Learning}} via {{Matrix Factorization}}},
  author={Bellet, Aur{\'e}lien and Cyffers, Edwige and Frey, Davide and Gaudel, Romaric and Ler{\'e}v{\'e}rend, Dimitri and Ta{\"i}ani, Fran{\c c}ois},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Features
- Workload generation for decentralized learning simulations
- Experiments on Housing and Femnist datasets
- Optimal noise correlation computation
- Accounting experiments
- Visualization tools

## Prerequisites
- Python 3.12 (only tested version)
- Recommended: virtual environment (`python3.12 -m venv venv`)

## Installation

1. Install the package in development mode (recommended):
    ```bash
    pip install -e .
    ```

   This will automatically install all dependencies from `requirement.txt` and make the `MFDL` package importable.

   Alternatively, install dependencies separately:
    ```bash
    pip install -r requirement.txt
    ```

## Usage
By default, all workload matrices are cached in the `cache/` directory to save computation time. Pre-computed matrices are included with this submission.
This allows to skip expensive steps, especially for housing and large graphs.
To regenerate all workloads from scratch, delete the `cache/` directory and rerun the experiments.
Be warned this may take time and can be expensive in terms of memory.

## Reproducing experiments
### 1. Accounting Experiments (Figure 2)

Run:
```bash
python -m MFDL.simulations.muffliato_accounting
```

### 2. Housing Experiments (Figure 3, top row)
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
python -m MFDL.simulations.tradeoff_plotter --dataset <dataset_name>
```
with `dataset_name` being either `housing` or `femnist`.
This will generate the corresponding figures under `figures/`.

### 3. Femnist Experiments (Figure 3, bottom row)
Follow the same steps as housing, but replace `experiments_housing.sh` with `experiments_femnist.sh`, and only use the graphs `florentine` and `ego`.
We also provide an additional way to submit SLURM jobs for each experiments instead, but require both a SLURM cluster and a custom

### 4. Optimal Workload Experiments (optional)

This experiment is optional and not included in the paper, but allows visualizing how correlation approaches affect the surrogate loss defined in the paper, validating that our approach indeed minimizes it.
Run:
```bash
python -m MFDL.simulations.factorization_experiments
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