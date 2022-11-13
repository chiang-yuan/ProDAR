# ProDAR

ProDAR enhances protien function prediction and extracts Dynamically Activated Residues (DARs) using the dynamical information obtained from normal mode analysis (NMA). The code is published with [Encoding protein dynamic information in graph representation for functional residue identification](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(22)00261-2). 

[[arXiv]](https://arxiv.org/abs/2112.12033) [[CRPS]](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(22)00261-2)

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
  where `${TORCH}` and `${CUDA}` should be repalced by the PyTorch and CUDA version (`TORCH=1.10.0` and `CUDA=cu113` for this specific environment).

3. Extra packages (if not installed by previous steps) may be installed via pip wheel.

## Data

To preprocess data and generate protein graphs, execute the first script to download raw data from [RCSB PDB](https://www.rcsb.org/) [search API](https://search.rcsb.org/) and [PDBe](https://www.ebi.ac.uk/pdbe/) [SIFTS API](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html), and execute the second script to export filtered PDB and GO entries as JSON graphs.

1. Execute `data-sifts.py` 
  ```
  python data-sifts.py
  ```
2. Execute `data-graphs.py`
  ```
  python data-graphs.py
  ```
> For the above two steps, `*.ipynb` files are provided for markdown and optional visualization when jupyter lab/notebook is used.

## Run

### Experiment (currently only k-fold cross validation)
```
python experiment_kfold.py <options>
```
### Evaluation (currently execute all saved models in `history/`)
```
python evaluation_kfold.py
```

## Citing
If you happen to use the scripts, analyses, models, results or partial snippet of this work and find it useful, please cite the associated paper
```Bibtex
@article{chiang2022encoding,
  title={Encoding protein dynamic information in graph representation for functional residue identification},
  author={Chiang, Yuan and Hui, Wei-Han and Chang, Shu-Wei},
  journal={Cell Reports Physical Science},
  volume={3},
  number={7},
  pages={100975},
  year={2022},
  publisher={Elsevier}
}
```

## License
TBD
