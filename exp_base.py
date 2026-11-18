"""E1/E2: base case (two fleets, N=50 SAA scenarios) solved by Clarabel, IPM-LU/CG/GMRES and
IPM-VQLS; in-sample AC validation and out-of-sample policy evaluation incl. baselines."""
import json, pickle, time, sys
import numpy as np, pandas as pd
import grid as g, core, ipm, ref, vqls
from model import Model

SEED = 47273; NS = 50; NTEST = 300
FL = g.FLEETS[:2]
sc = g.sample_scenarios(NS, SEED, FL)
out = {}
runs = {}
for name, solver in [('LU', ipm.lu_solver), ('CG', ipm.cg_solver), ('GMRES', ipm.gmres_solver), ('VQLS', None)]:
    if name == 'VQLS':
        V = vqls.VQLS(layers=12, seed=SEED)
        solver = lambda S, r, ctx: V.solve(S, r, ctx)
    log = []
    t0 = time.perf_counter()
    r = core.solve_case(FL, sc, reduced_solver=solver, log=log)
    wall = time.perf_counter() - t0
    runs[name] = dict(x=r['x'], Y=r['Y'], hist=r['hist'], log=log, wall=wall)
    m = r['model']
    bd = m.breakdown(r['x'], r['Y'])
    print(name, 'obj', r['hist'][-1]['obj'], 'newton', sum(h['newton'] for h in r['hist']), 'wall', wall, flush=True)
    if name == 'VQLS':
        runs[name]['vqls_records'] = V.records
    runs[name]['breakdown'] = bd
    runs[name]['viol'] = core.violations(r['ac'])
    if name == 'LU':
        # Clarabel reference on the final linearisation of the LU run
        t0 = time.perf_counter(); xr, Yr, pr = ref.reference(m); tref = time.perf_counter() - t0
        runs['CLARABEL'] = dict(x=xr, Y=Yr, obj=m.objective(xr, Yr), wall=tref, status=pr.status,
                                breakdown=m.breakdown(xr, Yr))
        print('CLARABEL', runs['CLARABEL']['obj'], tref, flush=True)
        final_model = m
pickle.dump(dict(runs=runs, sc=sc), open('results/base_runs.pkl', 'wb'))
import os
if os.environ.get('SKIP_OOS'):
    print('done (OOS skipped)'); raise SystemExit

# ---------------------------------------------------------------- out-of-sample
test = g.sample_scenarios(NTEST, SEED + 1, FL)
T = g.T
# deterministic expected-value schedule
mean_sc = {k: (v.mean(axis=0, keepdims=True) if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
mean_sc['n'] = 1
rd = core.solve_case(FL, mean_sc)
x_det = rd['x']
# [38]-type prescribed V2G case: single fleet at Bus 18 with a prescribed profile
x38 = np.zeros(2 * T)
x38[[0, 1, 2, 3, 4, 5, 22, 23]] = -0.20
x38[[17, 18, 19, 20]] = 0.30
# uncoordinated charging on arrival (no discharge)
xunc = np.zeros(2 * T)
xunc[[17, 18, 19, 20]] = -0.12
xunc[T + np.array([8, 9, 10])] = -0.10
policies = {'Proposed (VQLS-IPM)': runs['VQLS']['x'], 'Proposed (LU-IPM)': runs['LU']['x'],
            'Deterministic EV': x_det, '[38]-type prescribed': x38, 'Uncoordinated charging': xunc}
rc = core.Recourse(FL)
oos = {}
for k, x in policies.items():
    t0 = time.perf_counter()
    rows, Yt, arr = rc.evaluate(x, test)
    oos[k] = dict(rows=rows, arrays=arr, x=x)
    d = pd.DataFrame(rows)
    print(k, 'oos obj', d.obj.mean(), 'max soft', d.soft.max(), 'vmin', d.vmin.min(), time.perf_counter() - t0, flush=True)
pickle.dump(dict(oos=oos, test_seed=SEED + 1), open('results/base_oos.pkl', 'wb'))
print('done')
