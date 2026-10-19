# Finite-state validation of the minimax limit

This experiment checks the paper's minimax identity on the 64 states of `{0,1}^6`. Two independent linear programs compute the attackability radius and the optimal verifier gap for five distribution pairs and Hamming radii `0,1,2,3`. A second calculation measures estimation error from sampled source and target histograms.

The experiment runs on CPU with Python 3.10+, NumPy, and SciPy's HiGHS linear programming solver.

Run these commands from the repository root:

```bash
python -m pip install numpy scipy

# Population calculation only: 20 pairs of independent linear programs.
python experiments/barrier_i_finite_state/run_experiment.py \
  --output-dir outputs/finite-state-population \
  --skip-plugin

# Paper settings: 20 repeats per setting and sample size.
python experiments/barrier_i_finite_state/run_experiment.py \
  --output-dir outputs/finite-state \
  --seed 20260723 \
  --plugin-sizes 250,1000,5000 \
  --plugin-repeats 20
```

The command above sets the paper's 20 repeats explicitly; the script defaults to 10. Results are saved under `--output-dir` as CSV tables, JSON and Markdown summaries, and the generated distributions.

The tests check agreement with the point-to-uniform closed form, equality of the independent LPs, and monotonicity as the allowed edit set expands:

```bash
python -m unittest discover \
  -s experiments/barrier_i_finite_state \
  -p 'test_*.py'
```
