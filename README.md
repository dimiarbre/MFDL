# MFDL: Matrix Factorization for Decentralized Learning

## Installation
The only tested python version is python3.12. Once in an environment, run

```bash
pip install -r requirement.txt
```

## Running:
For the comparison between optimal workloads
```bash
python simulations/factorization_experiments.py
```

For the housing experiments, edit the configurations you want in `experiments_housing.sh` and then run it.
```bash
./experiments_housing.sh
```
Then, use `simulations/housing_plotter.py` to visualize the data you want.

## Datasets
* The Facebook ego graph can be downloaded here: https://snap.stanford.edu/data/ego-Facebook.html