# Chronos‑2 Evaluation

This directory provides uses Amazon's [Chronos‑2](https://github.com/amazon-science/chronos-forecasting) pretrained forecasting model to reproduce results from the mobility baselines experiments. The setup runs a sliding‑window evaluation over various public datasets.


## Repository layout

```
Chronos-2-evaluation/
├── conf/                # Hydra configuration
│   ├── experiment.yaml  # Base experiment settings
│   └── dataset/         # One YAML per dataset
├── data/                # Data files (HDF, CSV, etc.)
├── exp.sh               # Script to run the experiments
├── main.py              # Entry point that runs the evaluation
├── utils.py             # Dataset loaders, evaluation loop, helpers
├── pyproject.toml       # Python packaging / dependencies
└── UrbanEV/             # Specialized directory for this dataset
```

Each file in `conf/dataset` corresponds to one dataset (e.g. `SZ-taxi.yaml`). `main.py` merges the chosen dataset config with `conf/experiment.yaml` and launches the Chronos pipeline.


## Dependencies

We recommend creating a clean environment using **`uv` from Astral** (https://github.com/astral-sh/uv) via the `uv.lock` file:

```bash
uv python pin 3.12
uv sync
```


## Running experiments

The helper script (`exp.sh`) loops over a dataset list and invokes `main.py`:

Adjust `PY_EXEC` and the dataset array as required. Submit to your scheduler (e.g. `bsub`, `sbatch`) or run locally.

To run a single dataset directly:

```bash
python main.py dataset_cfg="dataset/NYC_Citi_Bike_pick-drop"
```


## Notes

* Data files are expected under `data/` with names matching the dataset configs. They can be downloaded from the mirror links provided in the main repository README. For UrbanEV, you need to place the data files under `UrbanEV/data` instead.
* `utils.set_seed` and `utils.load_dataset` handle reproducibility and loading; modify for custom preprocessing.
* To add a new dataset, add a YAML config to `conf/dataset` and ensure the loader in `utils` supports it.
* Any YAML field can be overridden from the command line interface. Hydra prints the merged config at startup so you can verify your settings.
