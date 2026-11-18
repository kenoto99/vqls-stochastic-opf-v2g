"""Record reduced Schur systems (S, r) along an LU-IPM run of the base case (N=50)."""
import numpy as np, pickle, grid as g, core, ipm
SEED = 47273
FL = g.FLEETS[:2]
sc = g.sample_scenarios(50, SEED, FL)
rec = []
def solver(S, r, ctx):
    rec.append((S.copy(), r.copy(), ctx['tau'], ctx['newton']))
    return np.linalg.solve(S, r), {}
log = []
r = core.solve_case(FL, sc, reduced_solver=solver, log=log)
pickle.dump(dict(rec=rec, log=log), open('results/snapshots.pkl', 'wb'))
print('recorded', len(rec))
