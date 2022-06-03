# ProDAR

## Hierarchy
```
├── data
│   ├── data-graphs.ipynb
│   ├── data-graphs.py
│   ├── data-sifts.ipynb
│   ├── data-sifts.py
│   ├── graphs-10A
│   ├── nma-anm
│   ├── pdbs
│   ├── pis
│   └── sifts
│       ├── mf_go_codes-allcnt.dat
│       ├── mf_go_codes-thres-50.dat
│       ├── mf_go_codes-thres-50.npy
│       ├── pdb_chains.dat
│       ├── pdbmfgos-thres-50.json
│       ├── sifts-err-1.log
│       └── sifts-err-2.log
├── datasets
│   └── dataset.py
├── evaluation_kfold.py
├── experiment_kfold.py
├── models
│   └── multilabel_classifiers
│       ├── GAT.py
│       ├── GCN.py
│       └── GraphSAGE.py
├── prodar-env.yml
└── prodar.py
```

## Environment

1. Clone environment from `prodar-env.yml` using [miniconda](https://docs.conda.io/en/latest/index.html):
```bash
conda env create -f environment.yml
```
2. Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) package via pip wheel:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
where `${TORCH}` and `${CUDA}` should be repalced by the PyTorch and CUDA version (`TORCH=1.10.0` and `CUDA=cu113` for this specific environment)
3. Extra packages (if not installed by previous steps) may be installed via pip wheel

## Data

To preprocess data and generate protein graphs, execuate the first script to download raw data from [RCSB PDB](https://www.rcsb.org/) [search API](https://search.rcsb.org/) and [PDBe](https://www.ebi.ac.uk/pdbe/) [SIFTS API](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html), and execuate the second script to export filtered PDB and GO entries as JSON graphs.

1. Execute `data-sifts.py` 
  ```
  python data-sifts.py
  ```
2. Execute `data-graphs.py`
  ```
  python data-graphs.py
  ```
> For above two steps, `*.ipynb` files are provided for markdown and optional visualization when jupyter lab/notebook is used.

## Run

### Experiment (currently only k-fold cross validation)
```
python experiment_kfold.py <options>
```
### Evaluation (currently execuate all saved models in `history/`
```
python evaluation_kfold.py
```
