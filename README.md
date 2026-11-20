# Publication folder

Everything shipped with the paper: the input data, the notebooks that make
synthetic datasets, fit the models and draw the figures, the saved results,
and the figures themselves. One folder per example.

```
publication/
  README.md
  figstyle.py          shared figure style (numpy + matplotlib only)
  diagnostics.py       critical points, ensembles, Surfe helpers, nets (torch optional)
  exploded3d.py, render_exploded.py   exploded 3-D scenes, rendered with headless Blender
  settings_table.py    writes the supplement's settings table from the saved fit settings
  <NN_example>/        one folder per main-text example
  s<NN_topic>/         one folder per Supplementary Note that needs its own figures
    *.ipynb            data, fitting and figure notebooks for this example
    data/              inputs: field or digitised data, or a dataset a notebook here makes
    results/           saved fits (grids, scalars, critical points, meshes, model weights)
    figures/           this example's figures (pdf + png); figures/supplement/ where the supplement has its own
    renders/           PNG renders and the camera used to compose the 3-D figures
```

Blender scene files (`*.blend`, `*.blend1`) are left out of the repository.
Re-running a 3-D figure notebook writes a new one locally. The Hartland data-panel
scene `05_hartland_quay/renders/data/scene/meshes.npz` is left out too: it is a
copy of the full photogrammetry cloud, and `05e` rebuilds it from
`HartlandQuay_Devon.ply`. The PNG and `camera.json` stay. `02_recumbent_pericline/`
may sit beside these folders in a working copy; it is left out of the repository too.

## Conventions

- **Run every notebook from its own folder** (Jupyter does this by default).
  Paths are relative to it: `data/`, `results/`, `figures/`, `renders/`.
- **Shared modules** live here, one level up. Each notebook's setup cell adds
  this folder to the import path before importing them:
  `sys.path.insert(0, str(Path("..").resolve()))`.
- **Synthetic datasets** are made by a notebook in the example's folder and
  written to its `data/`. Field and digitised data sit in `data/` as given.
- **Fitting notebooks** compute and show quick diagnostics that are *not*
  saved, and write to `results/`. The first fit also saves the evaluation grid
  (`inputs.npz`), so other methods, e.g. Surfe, run on exactly the same grid.
- **Figure notebooks** read `results/` and write `figures/`. The drawn panels
  need numpy and matplotlib. The 3-D panels also need PyVista, Pillow and
  Blender (`exploded3d.BLENDER`). The contact-distance cell at the end of
  `04c_multiwavelength_fold_figures` reloads the saved fits and needs curlew.
- **Kernels:** curlew notebooks run in `curlew` (below). The kernel is
  registered as `curlew_dev`, the name stored in the notebooks. Surfe notebooks
  run in `surfe`, with `surfepy` built at `DevOps/surfe/build`.
- **`settings_table.py`** reads the `fit_settings*.json` of the six main-text
  fits and writes `paper_tex/tables/settings.tex`. `paper_tex/latexmkrc` runs
  it before every build.

## Environment

`environment.yml` creates an environment named `curlew`, with the packages
these notebooks import. The versions are those installed in the author's
`curlew_dev` (macOS arm64; torch from pip). Create it and install curlew
from its own checkout (curlew is not on PyPI). The kernel is registered as
`curlew_dev`, matching the name stored in the notebooks:

```
conda env create -f environment.yml
conda activate curlew
pip install -e /path/to/curlew
python -m ipykernel install --user --name curlew_dev
```

The env provides Python 3.11 and:

| package | used for |
|---|---|
| numpy, torch, tqdm | curlew |
| matplotlib, cmasher | figures |
| pandas | the fold-pair contacts |
| scipy, contourpy, scikit-image | grids, contours, marching cubes |
| pyvista, pillow | the 3-D renders and compositing them |
| ipykernel | running the notebooks |

Blender is a separate application. The Surfe notebooks are not in this env.

## Examples

| folder | data | fitting | figures |
|---|---|---|---|
| `01_recumbent_fold` (2-D) | made by `01a_recumbent_fold_data` (with `synthetic_fold.py`) → `data/synthetic/` | `01b_recumbent_fold_fit` (one NF and one LDI fit, scored against the true units), `01c_recumbent_fold_surfe`, `01d_recumbent_fold_ensemble` (15 Fourier seeds, each member scored against the true units) → `results/synthetic/` | `01e_recumbent_fold_figures` (the comparison and the ensemble; filenames end in `_synthetic`) |
| `03_single_layer` (2-D, no Surfe) | made by `03a_single_layer_data` from `data/single_layer_digitised.svg` → `data/single_layer_ground_truth.npz`, `data/single_layer_observations.npz` | `03b_single_layer_fit` (writes `results/fit_settings.json`), `03c_single_layer_ensemble` (three ensembles of 15: no energy bias, energy bias, energy bias and strain) → `results/`, `03e_single_layer_strain_prior` → `results/strain_prior.npz`; shared through `single_layer_inputs.py` | `03d_single_layer_figures` (the main figure, and the numbers quoted in the text), `03f_single_layer_strain_prior_figures` |
| `04_multiwavelength_fold` (2-D, no Surfe) | `data/big_fold_contacts.csv` (ImplicitBenchmark `BigFoldDigitized.svg`, also in `data/`); `04a_multiwavelength_fold_data` → `data/big_fold_observations.npz` | `04b_multiwavelength_fold_fit` (energy-biased LDI, shear and RK4 one after the other) → `results/shear/`, `results/rk4/`; shared through `big_fold_inputs.py` | `04c_multiwavelength_fold_figures` (both integrators on one figure; `INTEGRATOR` picks which fit the printed numbers use) |
| `05_hartland_quay` (Devon outcrop) | `data/raw/` (traces, bedding, photogrammetry); `05a_hartland_quay_data` → `data/processed/` (read through `hartland_inputs.py`), and the supplement data and stereonet figures | `05b_hartland_quay_ldi` (20 Fourier seeds) → `results/`, `05c_hartland_quay_surfe` (Surfe, its meshes, and the block hold-out), `05d_hartland_quay_meshes` (member 19), `05i_hartland_quay_blocks` | `05e_hartland_quay_figures` (the member-19 and Surfe terrace renders, and the block figure), `05f_hartland_quay_finalcomp` (the composed paper figure), `05h_hartland_quay_numbers` (the numbers quoted in the text) |
| `06_claudius` (3-D seismic) | `data/` (Claudius, ImplicitBenchmark; see `data/_README.md`); `06a_claudius_data` → `data/processed/` (read through `claudius_inputs.py`) and the supplement data figure | `06b_claudius_ldi` and `06c_claudius_surfe`, both with `SAMPLING = "seismic"` and `LINES` of 5 by 5 plus the bounds, half-width 20 m, saved as `seis5x5b_w20` | `06d_claudius_figures` (`MODELS` names those two saved meshes) |
| `07_cylindrical_isocline` (3-D) | `data/raw/` (the ImplicitBenchmark refolded isocline); `07a_cylindrical_isocline_data` → `data/processed/` (read through `isocline_inputs.py`) | `07b_cylindrical_isocline_ldi` (`cylindrical=True`, reference facing −x via `flat_reference.py`), `07c_cylindrical_isocline_surfe` (`ANISOTROPY = 1e5`, saved as `surfe_aniso100000`), `07d_cylindrical_isocline_meshes`, `07e_cylindrical_isocline_ensemble` (15 Fourier seeds) | `07f_cylindrical_isocline_figures` (`SURFE_TAG = "surfe_aniso100000"`) |
| `s01_fourier_basis` (Supplementary Note 1) | the recumbent fold of `01_recumbent_fold` (its data and `fit_settings.json`) | `s01a_fourier_basis_directions` → `results/direction_seeds.npz`; `s01b_fourier_basis_sweeps` → `results/band_sweep.npz`; `s01c_fourier_basis_energy_bias` → `results/energy_bias.npz` | `s01d_fourier_basis_figures` |
| `s02_velocity` (Supplementary Note 2) | the recumbent fold of `01_recumbent_fold` (its data and `fit_settings.json`) | `s02a_velocity_fit` → `results/velocity.npz` | `s02b_velocity_figures` |
| `s03_shears` (Supplementary Note 3) | made by `s03a_shears_data` from `data/DisharmonicFolds.svg` → `data/disharmonic_fold.npz` | `s03b_shears_fit` → `results/shears.npz`; `s03c_shears_ensemble` → `results/shears_ensemble.npz` | `s03d_shears_figures` |

The fold-pair figures are `fig01_recumbent_fold_synthetic` and
`fig02_recumbent_fold_ensemble_synthetic`.

Surface meshes are saved as plain arrays (`diagnostics.export_meshes`), so the
3-D figures need no field evaluation. The Hartland figure is the composed
render of member 19 and of Surfe, cut back from the wave-cut platform to the
cliff (`05f`, from the PNGs written by `05e`). The isocline figure is the
slab renders of the truth, Surfe at anisotropy 1e5, and LDI. The Claudius
figure opens the `seis5x5b_w20` meshes.

## Outside this folder

`Paper/figstyle.py`, `Paper/diagnostics.py`, `Paper/exploded3d.py` and
`Paper/render_exploded.py` are symlinks to the copies here, and the data files
that moved out of `Paper/data/` are symlinked back, so the notebooks left in
`Paper/` keep working. The symlinks are absolute paths on the author's machine
and are not part of the shipped folder.
