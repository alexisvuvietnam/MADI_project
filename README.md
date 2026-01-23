# MADI Project: Implementation of FCI (Fast Causal Inference) Algorithm

Hello! This is the README file for the 2025/26 MADI project, part of the AI2D program at Sorbonne University.

**Authors**:
- Hoang Nguyen VU - 21107598
- Inés Tian RUIZ-BRAVO PLOVINS - 28705881

## Table of content

1. [Overview](#instruction)
2. [Requirements](#req)
3. [Structure](#file)

### Overview<a name="instruction"></a>

In this project, we study learning algorithms for causal inference and implement a pseudo-version of the FCI algorithm along with its variant RFCI. The implementation relies on several libraries for computing marginal and conditional independencies and for graph visualization.

All necessary classes and functions are contained in `FCI.py`. In particular, the functions `run_FCI` and `run_RFCI` correspond to the two available algorithms.

A `run_XXX` function takes the following main input parameters:
- `df: pandas.DataFrame`: The Pandas DataFrame containing the learning dataset.
- `alpha: float`: The significance level (p-value threshold) used to assess independence.
- `useGum: bool`: A boolean parameter, set to False by default:
    - If `useGum` is set to `True`, the method `pyagrum.BNLearner.G2` is used to compute independencies.
    - If `useGum` is set to `False`, the function `pingouin.partial_corr` is used to compute independencies.
- `bayesnet: Optional[pyagrum.BayesNet]`: The Bayesian network associated with the input dataset, if available.

Experimental analyses for the empirical study are implemented in the Python notebook `test.ipynb`.

### Requirements<a name="req"></a>

```text
python>=3.10
pyagrum>=2.3
numpy>=2.2
causallearn>=0.1
pandas>=0.5
bnlearn>=0.12
graphviz>=0.21
IPython>=9.9
```

### Structure<a name="file"></a>

```bash
.
├─ FCI.py
├─ test.ipynb
├─ README.md
```
