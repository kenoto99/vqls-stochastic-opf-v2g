# VQLS-assisted two-stage stochastic OPF for V2G-integrated distribution feeders

Reproduction code for the manuscript "A VQLS-Based Hybrid Quantum-Classical Framework for Dynamic Multi-Period Stochastic Optimal Power Flow in V2G-Integrated Smart Grids" by S. Ninlawat, P. Prabpal, P. Khompittaya, S. Sonasang, K. Phanrattanachai and N. Angkawisittpan (submitted to IEEE Access).

Every figure and table in the manuscript is produced by the scripts below from the result files in `results/`. The quantum circuits are emulated classically (state vector and density matrix); no quantum hardware was used.

Python 3.13, NumPy 2.5.3, SciPy 1.18.1, CVXPY 1.9.3 with Clarabel 0.11.1, pandas, matplotlib.
Install: `pip install numpy scipy cvxpy clarabel pandas matplotlib`

| File | Content |
|---|---|
| grid.py | IEEE 33-bus (Baran-Wu, unscaled) and 69-bus feeder data, AC backward-forward sweep, LinDistFlow, fleets, Monte Carlo scenarios |
| model.py | Two-stage SAA model (first stage: fleet schedule; recourse: OLTC, reactive power, deviation) |
| ipm.py | Barrier interior-point method with Schur reduction onto the first stage; LU/CG/GMRES solvers |
| vqls.py | VQLS: Pauli decomposition, QWC grouping, ansatz, global cost, rescaling, shots, density-matrix noise, ZNE |
| ref.py | Clarabel reference and Phase I |
| core.py | Successive linearization with AC corrections, AC validation, out-of-sample recourse |
| exp_*.py | Experiments (base case, snapshots, VQLS statistics, SAA, weights, scaling, noise in the loop, PF-only, depth) |
| fig_*.py, gen_numbers.py | Figures and LaTeX tables/macros generated from results/ |

Run order: exp_base.py, exp_pfonly.py, exp_snap.py, exp_stats.py, exp_saa.py, exp_pareto.py, exp_scaling.py,
exp_noiseloop.py, exp_depth7.py, exp_fleets.py, exp_rem.py, exp_ext_case.py (args: 33 48 21 and 69 24 12),
exp_ext_oos.py (same network and step arguments), then fig_*.py and gen_numbers.py (set PAPER_DIR for the table output).
exp_pareto.py uses Clarabel as the inner solver (see Section V-G). ieee69.npz holds the IEEE 69-bus data from MATPOWER case69. Seeds: 47273 (in-sample), 47274 (test).
Total run time on 2 CPU cores was roughly 5 hours.

## Data sources
IEEE 33-bus: M. E. Baran and F. F. Wu, IEEE Trans. Power Del., 1989. IEEE 69-bus: MATPOWER `case69` (R. D. Zimmerman et al., IEEE Trans. Power Syst., 2011).

## License
MIT (see LICENSE).
