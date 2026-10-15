# RRL_PowerSpectra
Calculates the auto-power spectrum for RRL contamination to the 21cm signal, and the cross-power spectra.

## Code

| File | Contents |
| --- | --- |
| `RRL_PS_Model.py` | Power spectrum model: 21cm, RRL, and cross terms, biases, bandwidth convolution, `PowerScaling` |
| `lineTemp.py` | RRL line frequencies, mean brightness temperature, 21cm temperature, optical depth helpers |
| `tauCalc_model.py` | Hydrogen and carbon RRL optical depth models (`tau_out`, `carbonModel`) |
| `growthFactorCalc.py` | Linear growth factor D+(z) |

| Notebook | Purpose |
| --- | --- |
| `RRL_PS_calculator.ipynb` | Generates and plots the RRL / 21cm power spectra, shot noise |
| `RRL_cosmology_calculator.ipynb` | Effect of RRL contamination on cosmology (BAO shift) |
| `RRL_tau_Model.ipynb` | Optical depth models compared with observations |
| `Unmatching_tempCurves.ipynb` | RRL line temperature curves, model vs. analytical, carbon estimate |

`legacy/` holds the earlier version of the power spectrum code.

Notebooks are committed with outputs cleared; run them to regenerate the figures.

## Requirements
Python 3 with `numpy`, `scipy`, `matplotlib`, `mcfit`, `tqdm`, `pathos`.

## Input data (not included)
Data files are not tracked in this repository. The code reads them by relative path, so run it from the repository root with this layout:

```
initSB_matterpower_z0.dat          # z=0 matter power spectrum
bias_21.txt  bias_rrl.txt          # 21cm and RRL bias vs. redshift
Omega_HI.txt                       # Omega_HI(z)
carbon_tau_combined.txt            # carbon RRL optical depth data
models/
  bn_hrrl_v2.npy  beta_hrrl_v2.npy  bn_hrrl_head_v2.p   # hydrogen b_n, beta_n departure coefficients
  bbeta_carbon*.txt  carbon_ne005_te100.txt              # carbon b_n beta_n tables
  crrl_bn_beta_T85K_n80-500.dat
  inttau_carbon*.txt
P_RRL_data/                        # precomputed power spectra (PS_z*_bw*_kperp*.npz), output of RRL_PS_calculator
```

Figures are written to `figures/`.
